"""derive_confidence and derive_score_confidence: pure Python, no model needed.

A non-finite probability means the forward pass produced garbage. It must read as "no
confidence", never as certainty: the documented use of confidence is to decide what to act on
automatically.
"""

from __future__ import annotations

import math

import pytest

from strands_decider.schema import derive_confidence, derive_score_confidence

NAN, INF = math.nan, math.inf


@pytest.mark.parametrize("derive", [derive_confidence, derive_score_confidence])
@pytest.mark.parametrize(
    "probs",
    [
        [NAN, 0.5, 0.5],
        [0.5, NAN, 0.5],
        [0.5, 0.5, NAN],
        [NAN, NAN],
        [INF, 0.0, 0.0],
        [0.0, -INF, 0.0],
        [0.2, 0.3, INF],
        [NAN],  # a single option is "forced", but not when its probability is garbage
    ],
)
def test_non_finite_input_has_no_confidence(derive, probs):
    assert derive(probs) == 0.0


def test_the_reported_case():
    # This returned 1.0: max() skips a NaN in the first position.
    assert derive_confidence([NAN, 0.5, 0.5]) == 0.0


def test_finite_input_is_unchanged():
    assert derive_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert derive_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert derive_confidence([0.5, 0.3, 0.2]) == pytest.approx((3 * 0.5 - 1) / 2)
    assert derive_confidence([0.7]) == 1.0
    assert derive_confidence([]) == 1.0
    assert derive_score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert derive_score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert derive_score_confidence([0.0, 0.0, 0.0]) == 0.0
    assert derive_score_confidence([1.0]) == 1.0
