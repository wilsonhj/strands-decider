"""A small synchronous client for a running decision-model server.

Deliberately `urllib` rather than a client library: these examples should need nothing
beyond `strands-agents`, so that a reader can drop one into their own project without
first agreeing to a dependency.

Plain HTTP and no authentication: this talks to a server you started yourself, on this
machine or a private network (`STRANDS_DECIDER_URL`). It does not sign requests, so it cannot
call a SageMaker endpoint (which needs AWS SigV4) or any server behind auth.

Synchronous on purpose too. Strands calls a handler method that is a plain `def`
directly, so a blocking request is the simplest thing that works -- and it keeps requests
to the server strictly sequential, which is how it is best driven: one request at a time.

**Renaming.** The product name is spelled out in exactly one place, `NAME` below, and the
things derived from it sit directly underneath. Everything else -- this module's filename,
the `Decider` class, the demos -- is named for what it does rather than what it is called,
so a future rename is an edit to this block and nothing else.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

# --------------------------------------------------------------------------- the name

NAME = "strands-decider"
"""What this thing is called today. The only place the name is written."""

ENV_URL = "STRANDS_DECIDER_URL"
"""Environment variable that overrides the endpoint. Derived from NAME, so rename together."""

SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
"""How to start a server, shown when nothing is listening.

This is the upstream CLI entry point (`[project.scripts]` in `pyproject.toml`), which is a
separate thing from NAME: it only changes if the Python package itself is renamed.
"""

MODEL_ID = "hobson-latest"
"""Sent as `model` on every request. The server ignores it and answers as whatever
checkpoint it loaded, so this is only here to show the full wire shape."""

DEFAULT_PORT = 8099
DEFAULT_URL = os.environ.get(ENV_URL, f"http://127.0.0.1:{DEFAULT_PORT}")


class ServerUnavailable(RuntimeError):
    """The server did not give a usable answer.

    One exception for every way that can go wrong -- nothing listening, a timeout, an HTTP
    error status (a request the server refused as too large or invalid is a 4xx), a reply
    that is not the expected JSON -- so a caller that must not act on a missing answer has a
    single thing to catch. Raised with the command that starts a server when none is there.
    """


class Decider:
    """One state, many typed questions, one round trip.

    The questions cannot see each other, so their answers stay independent -- but the
    state is encoded once and shared across them, which is why asking eight things costs
    far less than eight requests. Ask everything you need at once.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        # Generous: on Apple silicon the *first* request at a given input length pays a
        # one-off MPS shape compile, which is seconds on a long document.
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None

    # ---- questions -------------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true), and with two outcomes that *is* the confidence.

        `criteria` optionally describes what `"true"` and `"false"` mean. It is worth
        writing: a noul renders as a two-option list, so those descriptions are read the
        same way a choice's options are, and they sharpen the boundary considerably.
        """
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means.

        The option names are read from this request, not baked into the model, so you can
        change them without retraining anything.
        """
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ----------------------------------------------------------

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        return ServerUnavailable(
            f"no {NAME} at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers come back under the same keys."""
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
            answers = payload["answers"]
            if not isinstance(answers, dict):
                raise TypeError("`answers` is not an object")
        except urllib.error.HTTPError as exc:
            # The server is up and said no: keep its reason, do not suggest starting one.
            raise ServerUnavailable(
                f"{NAME} at {self.url} answered HTTP {exc.code}: {self._detail(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A read timeout is a bare TimeoutError (OSError), not a URLError, and a
            # truncated or non-JSON body is a ValueError.
            raise ServerUnavailable(
                f"{NAME} at {self.url} gave no usable answer ({type(exc).__name__}: {exc})"
            ) from exc
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return answers

    @staticmethod
    def _detail(exc: urllib.error.HTTPError) -> str:
        """The server's own explanation (FastAPI puts it under `detail`), kept short."""
        try:
            return str(json.loads(exc.read()).get("detail", exc.reason))[:300]
        except (OSError, ValueError, AttributeError):
            return str(exc.reason)

    @staticmethod
    def probability(answers: Mapping[str, Any], name: str) -> float:
        """P(true) of the noul answer `name`, checked.

        Raises ServerUnavailable if the answer is missing, is not a noul, or is not a
        finite number in 0..1, so a malformed reply can never read as a confident yes.
        """
        try:
            value = float(answers[name]["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ServerUnavailable(f"{NAME} returned no noul answer for {name!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ServerUnavailable(f"{NAME} returned {value!r} for {name!r}, not a probability")
        return value

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        h = self.health()
        return (
            f"{NAME}: {h['checkpoint']} on {h['base_model']} "
            f"[{h['device']}, window {h['max_length']}] at {self.url}"
        )

    def __str__(self) -> str:
        return f"{self.calls} decisions, {self.input_tokens} input tokens"
