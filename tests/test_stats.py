import numpy as np
import pytest

from doclens import stats


def test_rate_and_interval():
    rng = np.random.default_rng(0)
    chars = rng.integers(5, 40, 500)
    edits = rng.binomial(chars, 0.05)
    b = stats.bootstrap_rate(edits, chars, reps=500)
    assert b["rate"] == pytest.approx(edits.sum() / chars.sum())
    assert b["lo"] < b["rate"] < b["hi"] and b["hi"] - b["lo"] < 0.02


def test_paired_difference():
    rng = np.random.default_rng(1)
    chars = rng.integers(5, 40, 400)
    a = rng.binomial(chars, 0.03)
    b = a + rng.binomial(chars, 0.03)              # b makes every error a makes, and more
    same = stats.paired(a, a, chars, reps=500)
    assert same["diff"] == 0 and same["p"] == 1.0
    better = stats.paired(a, b, chars, reps=500)
    assert better["diff"] < 0 and better["hi"] < 0 and better["p"] < 0.01
    with pytest.raises(ValueError):
        stats.paired(a, b[:-1], chars)
