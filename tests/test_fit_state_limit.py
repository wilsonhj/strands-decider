"""`_fit` must not hand a huge state to the tokeniser while holding `_tok_lock`.

The engine is built without a model: `_fit` reads only the config, the tokeniser and the lock.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from strands_decider.infer import _MAX_CHARS_PER_TOKEN, EngineConfig, SystemOneEngine
from strands_decider.schema import RequestError

WINDOW = 64


class _Tok:
    """One token per character; records the longest text it was asked to encode."""

    def __init__(self, lock):
        self.lock = lock
        self.longest = 0
        self.locked_during = []

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kw):
        texts = [text] if isinstance(text, str) else list(text)
        self.longest = max(self.longest, *(len(t) for t in texts))
        self.locked_during.append(self.lock.locked())
        ids = [[1] * len(t) for t in texts]
        if kw.get("truncation"):
            ids = [x[: kw["max_length"]] for x in ids]
        out = {"input_ids": ids if not isinstance(text, str) else ids[0]}
        if return_offsets_mapping:
            out["offset_mapping"] = [[(i, i + 1) for i in range(len(t))] for t in texts]
        return out


def _engine(strict):
    eng = object.__new__(SystemOneEngine)
    eng.cfg = EngineConfig(device="cpu", strict_window=strict)
    eng.model = SimpleNamespace(config=SimpleNamespace(max_length=WINDOW))
    eng._tok_lock = threading.Lock()
    eng.tok = _Tok(eng._tok_lock)
    return eng


def test_huge_state_is_clipped_before_tokenising():
    eng = _engine(strict=False)
    state = "s" * 1_000_000
    s, q, _ = eng._fit(state, ["<question>\n1. a\n</question>\n<answer>"])
    assert eng.tok.longest <= WINDOW * _MAX_CHARS_PER_TOKEN
    assert all(eng.tok.locked_during)  # tokenisation still happens under the lock
    assert len(s) + len(q[0]) <= WINDOW  # and the window rule is unchanged


def test_a_state_that_fits_is_untouched():
    eng = _engine(strict=False)
    eng._fit("s" * 30, ["q" * 10])
    assert eng.tok.longest == 30


def test_strict_window_refuses_a_huge_state_without_tokenising_it():
    eng = _engine(strict=True)
    with pytest.raises(RequestError, match="context window"):
        eng._fit("s" * 1_000_000, ["q" * 10])
    assert eng.tok.longest <= 10  # only the (short) question was encoded
