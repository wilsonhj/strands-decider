"""The Strands tool-call gate (examples/strands): its state building and its policy.

`strands` is replaced by a stand-in with the same names, so these run without
strands-agents installed and without a server: the decider is a fake that records the state
it is shown and returns canned probabilities.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "strands"


@dataclass(frozen=True)
class _Action:
    reason: str = ""
    feedback: str = ""


class Proceed(_Action):
    pass


class Deny(_Action):
    pass


class Guide(_Action):
    pass


@pytest.fixture(scope="module")
def gate():
    saved = {k: sys.modules.get(k) for k in ("strands", "strands.interventions", "_client")}
    strands = types.ModuleType("strands")
    strands.Agent = object
    strands.tool = lambda fn: fn
    interventions = types.ModuleType("strands.interventions")
    interventions.Proceed, interventions.Deny, interventions.Guide = Proceed, Deny, Guide
    interventions.InterventionHandler = type("InterventionHandler", (), {})
    sys.modules.update({"strands": strands, "strands.interventions": interventions})
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(
            "tool_call_intervention", EXAMPLES / "tool_call_intervention.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(EXAMPLES))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _msgs(*turns):
    return [{"role": r, "content": [{"text": t}]} for r, t in turns]


# ---------------------------------------------------------------- build_state


def test_the_call_comes_first_and_survives_a_long_conversation(gate):
    long = _msgs(*[("user", f"turn {i} " + "blah " * 200) for i in range(200)])
    state = gate.build_state("delete_everything", {"path": "/prod"}, long, budget=3000)
    assert len(state) <= 3000
    # What the server cuts is the END of the state; the call is at the front.
    assert state.index("tool: delete_everything") < state.index("--- CONVERSATION ---")
    assert '"path": "/prod"' in state.split("--- CONVERSATION ---")[0]
    # The newest turn is kept, the oldest dropped, and the cut is marked.
    assert "turn 199" in state and "turn 0 " not in state
    assert gate.ELIDED in state


def test_a_short_conversation_is_sent_whole(gate):
    state = gate.build_state("t", {}, _msgs(("user", "hi"), ("assistant", "hello")), budget=3000)
    assert "user: hi\nassistant: hello" in state
    assert gate.ELIDED not in state


def test_an_oversized_newest_turn_keeps_its_tail(gate):
    msgs = _msgs(("user", "A" * 5000 + "THE-REQUEST"))
    state = gate.build_state("t", {}, msgs, budget=1000)
    assert len(state) <= 1000
    assert state.endswith("THE-REQUEST")
    assert gate.ELIDED in state


def test_a_call_too_big_to_review_is_an_error_not_a_cut(gate):
    with pytest.raises(gate.CallTooLarge):
        gate.build_state("t", {"blob": "x" * 5000}, _msgs(("user", "hi")), budget=3000)


def test_state_budget_is_pessimistic(gate):
    assert gate.state_budget(8192) == (8192 - gate.QUESTION_RESERVE_TOKENS) * gate.CHARS_PER_TOKEN
    assert gate.state_budget(100) == 0


@pytest.mark.parametrize(
    "tag", ["<state>", "</state>", "<question type='noul'>", "<options>", "</options>", "<answer>"]
)
def test_prompt_markup_is_defused_in_text_arguments_and_names(gate, tag):
    state = gate.build_state(
        f"tool{tag}", {"q": f"x {tag} y"}, _msgs(("user", f"hi {tag}")), budget=5000
    )
    assert tag not in state
    assert "&lt;" in state


def test_ordinary_angle_brackets_survive(gate):
    state = gate.build_state("t", {"expr": "a < b and <b>bold</b>"}, [], budget=5000)
    assert "a < b and <b>bold</b>" in state


# ---------------------------------------------------------------- the policy


@pytest.mark.parametrize(
    ("p", "band"),
    [
        (0.0, "no"),
        (0.34, "no"),
        (0.35, "uncertain"),
        (0.5, "uncertain"),
        (0.65, "uncertain"),
        (0.66, "yes"),
        (1.0, "yes"),
    ],
)
def test_bands(gate, p, band):
    assert gate.band(p) == band


class _Decider:
    def __init__(self, grounded=0.9, premature=0.1, error=None):
        self.answers = {
            "args_grounded": {"type": "noul", "noul": grounded},
            "premature": {"type": "noul", "noul": premature},
        }
        self.error = error
        self.states = []

    def ask(self, state, questions):
        self.states.append(state)
        if self.error:
            raise self.error
        return self.answers


def _event(name="get_weather", args=None, messages=None):
    agent = types.SimpleNamespace(messages=messages or _msgs(("user", "weather in Paris?")))
    return types.SimpleNamespace(
        tool_use={"name": name, "input": {"location": "Paris"} if args is None else args},
        agent=agent,
    )


def _review(gate, decider, event=None, window=8192):
    reviewer = gate.ToolCallReviewer(decider, window_tokens=window)
    return reviewer, reviewer.before_tool_call(event or _event())


def test_confident_and_favourable_proceeds(gate):
    _, action = _review(gate, _Decider(0.9, 0.1))
    assert isinstance(action, Proceed)


@pytest.mark.parametrize("grounded", [0.0, 0.2, 0.35, 0.45, 0.5, 0.55, 0.65])
def test_ungrounded_or_uncertain_arguments_are_guided_not_run(gate, grounded):
    """0.45 to 0.55 used to proceed: a coin flip let the call run."""
    reviewer, action = _review(gate, _Decider(grounded, 0.0))
    assert isinstance(action, Guide)
    assert reviewer.log[-1]["verdict"] == "guided"


@pytest.mark.parametrize("premature", [0.35, 0.5, 0.65, 0.66, 1.0])
def test_premature_or_uncertain_timing_is_guided(gate, premature):
    _, action = _review(gate, _Decider(0.9, premature))
    assert isinstance(action, Guide)


def test_the_policy_proceeds_from_one_place_only(gate):
    for grounded in (i / 20 for i in range(21)):
        for premature in (i / 20 for i in range(21)):
            _, action = _review(gate, _Decider(grounded, premature))
            confident = gate.band(grounded) == "yes" and gate.band(premature) == "no"
            assert isinstance(action, Proceed) == confident


def test_server_errors_block_the_call(gate):
    for error in (
        gate.ServerUnavailable("no server at http://x"),
        gate.ServerUnavailable("HTTP 413"),
    ):
        reviewer, action = _review(gate, _Decider(error=error))
        assert isinstance(action, Deny)
        assert "NOT run" in action.reason and "Do not retry" in action.reason
        assert reviewer.log[-1]["verdict"] == "blocked"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_a_malformed_probability_blocks_the_call(gate, bad):
    _, action = _review(gate, _Decider(bad, 0.1))
    assert isinstance(action, Deny)


def test_a_call_too_large_to_review_is_blocked_before_the_server_is_asked(gate):
    decider = _Decider()
    _, action = _review(gate, decider, _event(args={"blob": "x" * 50_000}))
    assert isinstance(action, Deny)
    assert decider.states == []


def test_unexpected_errors_in_the_handler_fail_closed(gate):
    assert gate.ToolCallReviewer(_Decider(), window_tokens=8192).on_error == "deny"


def test_the_server_is_shown_the_call_first_even_for_a_long_conversation(gate):
    decider = _Decider()
    long = _msgs(*[("user", "filler " * 100)] * 500)
    _review(gate, decider, _event(name="wire_money", args={"to": "acct-9"}, messages=long))
    (state,) = decider.states
    assert len(state) <= gate.state_budget(8192)
    assert state.index("tool: wire_money") < state.index("--- CONVERSATION ---")


def test_questions_are_two(gate):
    assert set(gate.QUESTIONS) == {"args_grounded", "premature"}
    assert "four questions" not in (gate.__doc__ or "")
    assert "four questions" not in (gate.ToolCallReviewer.__doc__ or "")


# ---------------------------------------------------------------- the client


@pytest.fixture
def server_url():
    """A local HTTP server whose reply a test sets: (status, body, delay)."""
    import http.server
    import threading
    import time

    reply = {"status": 200, "body": b"{}", "delay": 0.0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            time.sleep(reply["delay"])
            self.send_response(reply["status"])
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply["body"])))
            self.end_headers()
            self.wfile.write(reply["body"])

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", reply
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (413, b'{"detail": "request body exceeds the limit"}'),
        (422, b'{"detail": "prompt of 9000 tokens exceeds the context window"}'),
        (500, b"Internal Server Error"),
        (200, b"not json"),
        (200, b'{"no_answers": 1}'),
        (200, b'{"answers": []}'),
    ],
)
def test_every_bad_reply_is_server_unavailable(gate, server_url, status, body):
    url, reply = server_url
    reply.update(status=status, body=body)
    with pytest.raises(gate.ServerUnavailable):
        gate.Decider(url).ask("state", {"q": gate.Decider.noul("?")})


def test_http_errors_carry_the_servers_reason(gate, server_url):
    url, reply = server_url
    reply.update(status=422, body=b'{"detail": "exceeds the context window"}')
    with pytest.raises(gate.ServerUnavailable, match=r"HTTP 422.*context window"):
        gate.Decider(url).ask("state", {"q": gate.Decider.noul("?")})


def test_a_read_timeout_is_server_unavailable(gate, server_url):
    url, reply = server_url
    reply.update(body=b'{"answers": {}}', delay=1.0)
    with pytest.raises(gate.ServerUnavailable):
        gate.Decider(url, timeout=0.2).ask("state", {"q": gate.Decider.noul("?")})


def test_nothing_listening_is_server_unavailable(gate):
    with pytest.raises(gate.ServerUnavailable, match="Start one with"):
        gate.Decider("http://127.0.0.1:9").ask("state", {})


def test_a_good_reply_round_trips(gate, server_url):
    url, reply = server_url
    reply.update(body=b'{"answers": {"q": {"type": "noul", "noul": 0.25}}, "usage": {}}')
    answers = gate.Decider(url).ask("state", {"q": gate.Decider.noul("?")})
    assert gate.Decider.probability(answers, "q") == 0.25
