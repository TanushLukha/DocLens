"""How sure a score is: bootstrap intervals for an error rate, and paired comparisons.

A test set is one sample of the images a reader will meet. Drawing the same number of images
from the set with replacement, many times, and scoring each draw shows how much the rate moves
with the sample: the middle 95% of those rates is the interval reported next to every headline
number.

Two systems read the same images, so they are compared on the same draws (a paired bootstrap):
what varies between draws is mostly which images are hard, and that cancels in the difference.
The p value is how often the difference crosses zero, doubled for a two sided test.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _draws(n: int, reps: int, seed: int, chunk: int = 200):
    rng = np.random.default_rng(seed)
    done = 0
    while done < reps:
        k = min(chunk, reps - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def bootstrap_rate(edits: Sequence[int], chars: Sequence[int], reps: int = 2000, seed: int = 0,
                   level: float = 0.95) -> dict:
    """Corpus rate sum(edits) / sum(chars) with its bootstrap interval over images."""
    e, c = np.asarray(edits, np.float64), np.asarray(chars, np.float64)
    if len(e) == 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rates = np.concatenate([e[idx].sum(1) / np.maximum(c[idx].sum(1), 1) for idx in _draws(len(e), reps, seed)])
    a = (1 - level) / 2
    return {"rate": float(e.sum() / max(c.sum(), 1)), "lo": float(np.quantile(rates, a)),
            "hi": float(np.quantile(rates, 1 - a)), "n": int(len(e))}


def paired(edits_a: Sequence[int], edits_b: Sequence[int], chars: Sequence[int], reps: int = 2000,
           seed: int = 0, level: float = 0.95) -> dict:
    """Rate of A minus rate of B on the same images: the difference, its interval and a p value."""
    a, b, c = (np.asarray(v, np.float64) for v in (edits_a, edits_b, chars))
    if not (len(a) == len(b) == len(c)) or len(a) == 0:
        raise ValueError("the two systems must be scored on the same, non empty list of images")
    diffs = np.concatenate([(a[idx].sum(1) - b[idx].sum(1)) / np.maximum(c[idx].sum(1), 1)
                            for idx in _draws(len(a), reps, seed)])
    q = (1 - level) / 2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float((a.sum() - b.sum()) / max(c.sum(), 1)), "lo": float(np.quantile(diffs, q)),
            "hi": float(np.quantile(diffs, 1 - q)), "p": float(min(1.0, p)), "n": int(len(a))}


def bootstrap_f1(tp: Sequence[int], fp: Sequence[int], fn: Sequence[int], reps: int = 2000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Micro F1 over items (receipts) with its bootstrap interval: F1 = 2 TP / (2 TP + FP + FN),
    summed over the items of each draw."""
    t, p, n = (np.asarray(v, np.float64) for v in (tp, fp, fn))
    if len(t) == 0:
        return {"f1": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    f = np.concatenate([2 * t[i].sum(1) / np.maximum(2 * t[i].sum(1) + p[i].sum(1) + n[i].sum(1), 1)
                        for i in _draws(len(t), reps, seed)])
    q = (1 - level) / 2
    return {"f1": float(2 * t.sum() / max(2 * t.sum() + p.sum() + n.sum(), 1)), "lo": float(np.quantile(f, q)),
            "hi": float(np.quantile(f, 1 - q)), "n": int(len(t))}


def paired_f1(a: tuple[Sequence[int], Sequence[int], Sequence[int]], b: tuple[Sequence[int], Sequence[int], Sequence[int]],
              reps: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """F1 of A minus F1 of B on the same items, each given as (TP, FP, FN) per item: the
    difference, its interval and a two sided p value."""
    ta, pa, na = (np.asarray(v, np.float64) for v in a)
    tb, pb, nb = (np.asarray(v, np.float64) for v in b)
    if not (len(ta) == len(tb)) or len(ta) == 0:
        raise ValueError("the two systems must be scored on the same, non empty list of items")

    def f(t, p, n):
        return 2 * t.sum(-1) / np.maximum(2 * t.sum(-1) + p.sum(-1) + n.sum(-1), 1)

    d = np.concatenate([f(ta[i], pa[i], na[i]) - f(tb[i], pb[i], nb[i]) for i in _draws(len(ta), reps, seed)])
    q = (1 - level) / 2
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(f(ta, pa, na) - f(tb, pb, nb)), "lo": float(np.quantile(d, q)),
            "hi": float(np.quantile(d, 1 - q)), "p": float(min(1.0, p)), "n": int(len(ta))}
