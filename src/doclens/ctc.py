"""Turning the network's per column scores into text: greedy decoding and prefix beam search.

Both take log probabilities, a T x C array (T columns, C classes, class 0 the blank).

Greedy (best path) takes the most likely class in every column, merges repeats and drops blanks.
It is fast, but it judges one alignment: a text whose probability is spread over many alignments
can lose to a worse text that has one strong alignment.

Prefix beam search keeps the `beam` most probable texts so far, summing for each one the
probability of every alignment that spells it (ending in a blank, or not, kept apart because
"a" + "a" is one letter without a blank between and two letters with one). It returns several
candidates with their probabilities, which is what a language model needs to rescore them.
"""
from __future__ import annotations

import math

import numpy as np

from .alphabet import BLANK

NEG_INF = float("-inf")


def greedy(logp: np.ndarray) -> list[int]:
    best = logp.argmax(axis=1)
    out, prev = [], BLANK
    for c in best.tolist():
        if c != prev and c != BLANK:
            out.append(c)
        prev = c
    return out


def _lae(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) without overflow."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def beam_search(logp: np.ndarray, beam: int = 10, prune: float = -8.0) -> list[tuple[tuple[int, ...], float]]:
    """The `beam` best label sequences with their total log probability, best first.

    prune: classes below this log probability in a column (minus 8, a probability of about 0.0003) are not tried
    there; the blank is always tried. Most columns of a trained model are confidently blank,
    so this keeps the search to a few candidates per column.
    """
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}   # prefix: (ends in blank, does not)
    for row in logp:
        cands = [int(c) for c in np.flatnonzero(row > prune) if c != BLANK]
        pblank = float(row[BLANK])
        nxt: dict[tuple[int, ...], list[float]] = {}

        def slot(prefix):
            s = nxt.get(prefix)
            if s is None:
                s = nxt[prefix] = [NEG_INF, NEG_INF]
            return s

        for prefix, (pb, pnb) in beams.items():
            total = _lae(pb, pnb)
            s = slot(prefix)
            s[0] = _lae(s[0], total + pblank)                    # a blank: the text stays the same
            last = prefix[-1] if prefix else None
            for c in cands:
                p = float(row[c])
                if c == last:
                    s[1] = _lae(s[1], pnb + p)                   # the same letter again, no blank between: merged
                    e = slot(prefix + (c,))
                    e[1] = _lae(e[1], pb + p)                    # after a blank: a second letter
                else:
                    e = slot(prefix + (c,))
                    e[1] = _lae(e[1], total + p)
        ranked = sorted(nxt.items(), key=lambda kv: _lae(kv[1][0], kv[1][1]), reverse=True)[:beam]
        beams = {k: (v[0], v[1]) for k, v in ranked}
    out = [(k, _lae(pb, pnb)) for k, (pb, pnb) in beams.items()]
    return sorted(out, key=lambda kv: kv[1], reverse=True)


def path_logprob(logp: np.ndarray, labels: list[int]) -> float:
    """log P(labels | image) by the CTC forward algorithm: the exact sum over alignments.

    Used by the tests to check the beam search, and handy to score any given text.
    """
    ext = [BLANK]
    for c in labels:
        ext += [c, BLANK]
    s = len(ext)
    alpha = [NEG_INF] * s
    alpha[0] = float(logp[0, BLANK])
    if s > 1:
        alpha[1] = float(logp[0, ext[1]])
    for t in range(1, logp.shape[0]):
        new = [NEG_INF] * s
        for i in range(s):
            a = alpha[i]
            if i >= 1:
                a = _lae(a, alpha[i - 1])
            if i >= 2 and ext[i] != BLANK and ext[i] != ext[i - 2]:
                a = _lae(a, alpha[i - 2])
            new[i] = a + float(logp[t, ext[i]]) if a != NEG_INF else NEG_INF
        alpha = new
    return _lae(alpha[-1], alpha[-2]) if s > 1 else alpha[-1]
