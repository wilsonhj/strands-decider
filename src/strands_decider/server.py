"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.

Limits. The server bounds what one request can cost, so a single caller cannot exhaust
memory, tokeniser time or the GPU. Each has a default, an argument to `create_app` /
`serve`, and an environment variable (the argument wins):

    STRANDS_DECIDER_MAX_BODY_BYTES     16 MiB   request body, checked while it streams
    STRANDS_DECIDER_MAX_QUESTIONS      64       questions in one request
    STRANDS_DECIDER_MAX_STATE_CHARS    500,000  the state as it is rendered into the prompt
    STRANDS_DECIDER_MAX_QUESTION_CHARS 50,000   one question as rendered (instructions + options)
    STRANDS_DECIDER_MAX_IMAGES         8        images in one request
    STRANDS_DECIDER_CONCURRENCY        1 on MPS, otherwise 2    forward passes at once

A request over a size limit gets 413. Requests beyond the concurrency limit wait their turn;
they are not rejected. The server has no authentication: bind it to a private interface or
put it behind something that authenticates (SageMaker's own SigV4, a gateway).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .prompting import render_content, render_question
from .schema import RequestError, SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None

ENV_PREFIX = "STRANDS_DECIDER_"


@dataclass(frozen=True)
class Limits:
    """What one request may cost. See the module docstring for the environment variables."""

    max_body_bytes: int = 16 * 1024 * 1024
    max_questions: int = 64
    max_state_chars: int = 500_000
    max_question_chars: int = 50_000
    max_images: int = 8

    @classmethod
    def resolve(
        cls,
        *,
        max_body_bytes: int | None = None,
        max_questions: int | None = None,
        max_state_chars: int | None = None,
        max_question_chars: int | None = None,
        max_images: int | None = None,
    ) -> Limits:
        """Each limit: the argument, else its environment variable, else the default."""
        given = {
            "max_body_bytes": max_body_bytes,
            "max_questions": max_questions,
            "max_state_chars": max_state_chars,
            "max_question_chars": max_question_chars,
            "max_images": max_images,
        }
        defaults = asdict(cls())
        return cls(**{k: _setting(k, v, defaults[k]) for k, v in given.items()})


def _setting(name: str, value: int | None, default: int, *, minimum: int = 1) -> int:
    """`value`, else $STRANDS_DECIDER_<NAME>, else `default`; must be at least `minimum`."""
    env = ENV_PREFIX + name.upper()
    if value is None:
        raw = os.environ.get(env, "").strip()
        try:
            value = int(raw) if raw else default
        except ValueError:
            raise ValueError(f"{env} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Answer 413 to a request body over `max_bytes`, without reading the rest of it.

    Checks the declared Content-Length up front, and counts the bytes as they arrive so a
    chunked body (no Content-Length) or a lying one is stopped as well. Whatever the app
    does once the limit is hit -- FastAPI turns a failed body read into a 400 -- the client
    sees the 413.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _refuse(self) -> JSONResponse:
        return JSONResponse(
            {"detail": f"request body exceeds the limit of {self.max_bytes} bytes"},
            status_code=413,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse()(scope, receive, send)
            return

        received = 0
        exceeded = False
        responded = False  # the app has started a response that is not ours

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal responded
            if exceeded:
                # The app is answering a body we cut off (as a 400, typically). Replace
                # its response with the 413, once, and drop the rest of it.
                if not responded:
                    responded = True
                    await self._refuse()(scope, receive, send)
                return
            if message["type"] == "http.response.start":
                responded = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            if not responded:
                await self._refuse()(scope, receive, send)
            else:  # pragma: no cover - the app began answering before reading the body
                raise


def check_limits(request: SystemOneRequest, limits: Limits) -> None:
    """Refuse (413) a request bigger than `limits` allow, before any of it is tokenised."""

    def too_big(what: str, n: int, cap: int) -> HTTPException:
        return HTTPException(status_code=413, detail=f"{what}: {n:,} exceeds the limit of {cap:,}")

    if len(request.questions) > limits.max_questions:
        raise too_big("questions", len(request.questions), limits.max_questions)
    if len(request.images) > limits.max_images:
        raise too_big("images", len(request.images), limits.max_images)
    state_chars = len(render_content(request.state))
    if state_chars > limits.max_state_chars:
        raise too_big("state characters", state_chars, limits.max_state_chars)
    for name, question in request.questions.items():
        n = len(render_question(question).text)
        if n > limits.max_question_chars:
            raise too_big(f"characters in question {name!r}", n, limits.max_question_chars)


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion (issue #9) when
    two overlapping threads hit the same device. CUDA and CPU are thread-safe for
    inference, and MLX serialises on its own scheduler."""
    return str(device).startswith("mps")


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    max_concurrency: int | None = None,
    max_body_bytes: int | None = None,
    max_questions: int | None = None,
    max_state_chars: int | None = None,
    max_question_chars: int | None = None,
    max_images: int | None = None,
) -> FastAPI:
    global _engine

    limits = Limits.resolve(
        max_body_bytes=max_body_bytes,
        max_questions=max_questions,
        max_state_chars=max_state_chars,
        max_question_chars=max_question_chars,
        max_images=max_images,
    )
    # Forward passes in flight at once, on every device. MPS aborts the process when two
    # overlap (issue #9), so it gets one; elsewhere a couple keep the GPU busy while another
    # request tokenises, and a flood of requests queues here instead of all running at once.
    concurrency = _setting("concurrency", max_concurrency, 1 if _needs_serialisation(device) else 2)

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=limits.max_body_bytes)

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        strict_window=strict_window,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    if vision:
        if device == "mlx":
            raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
        from .vision import load_vision_engine

        _engine = load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
    elif device == "mlx":
        _engine = load_mlx(checkpoint, config)
    else:
        model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
        _engine = SystemOneEngine(model, config)

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it.
    @app.get("/health")
    @app.get("/ping")
    def health() -> dict:
        eng = get_engine()
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            # The folder or repo name only, like `model`: enough to tell servers apart
            # without publishing the server's filesystem layout to every caller.
            "checkpoint": os.path.basename(checkpoint.rstrip("/")) or checkpoint,
            "base_model": eng.model.config.base_model,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
            "concurrency": concurrency,
            "limits": asdict(limits),
        }

    # Handlers are plain `def`, so each request runs on one of the threadpool's threads (40
    # by default). Without this, all of them could be inside a forward pass at once.
    slots = threading.BoundedSemaphore(concurrency)

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        check_limits(request, limits)
        started = time.perf_counter()
        try:
            with slots:
                response = eng.evaluate(request)
        except RequestError as exc:
            # The caller can fix this by changing the request. Any other exception,
            # ValueError included, is a fault in the server and surfaces as a 500 rather
            # than being blamed on the caller.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    max_concurrency: int | None = None,
    max_body_bytes: int | None = None,
    max_questions: int | None = None,
    max_state_chars: int | None = None,
    max_question_chars: int | None = None,
    max_images: int | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
        max_concurrency=max_concurrency,
        max_body_bytes=max_body_bytes,
        max_questions=max_questions,
        max_state_chars=max_state_chars,
        max_question_chars=max_question_chars,
        max_images=max_images,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
