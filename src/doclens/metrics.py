"""Character and word error rates, written out, with the alignment behind them.

CER = (substitutions + deletions + insertions) / characters in the reference, counted in
Unicode code points after `normalize`. A Devanagari conjunct such as क्ष is three code points
(क, virama, ष), so one wrong conjunct can cost up to three errors: the same convention
the Indic OCR papers use, and the reason a Hindi CER is not directly comparable to an
English one.

Corpus CER sums the edits over all samples and divides by the summed reference length,
so long lines weigh more than short ones. A mean of per sample rates hides a different
failure: a model that loops on one image can produce thousands of extra characters and
dominate the mean. So every score also reports the median and the catastrophic rate
(samples with CER above 0.5), as a published Devanagari OCR stress test recommends.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

from .text import normalize, wrong_script

CATASTROPHIC = 0.5


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance by dynamic programming with two rows: O(len(a) * len(b)) time, O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1,             # delete x
                           cur[j - 1] + 1,          # insert y
                           prev[j - 1] + (x != y)))  # keep or substitute
        prev = cur
    return prev[-1]


def align(ref: Sequence, hyp: Sequence) -> list[tuple[str, object, object]]:
    """One optimal alignment as (op, ref item, hyp item) with op in ok, sub, del, ins.

    Used for error analysis: which characters an engine confuses, drops or invents.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


@dataclass
class Sample:
    char_edits: int
    chars: int
    word_edits: int
    words: int
    wrong_script: bool = False

    @property
    def cer(self) -> float:
        return self.char_edits / max(self.chars, 1)


@dataclass
class Scores:
    """Accumulates samples; `summary()` gives the numbers every results table reports."""
    samples: list[Sample] = field(default_factory=list)

    def add(self, ref: str, hyp: str) -> Sample:
        r, h = normalize(ref), normalize(hyp)
        s = Sample(levenshtein(r, h), len(r), levenshtein(r.split(), h.split()), len(r.split()), wrong_script(r, h))
        self.samples.append(s)
        return s

    def summary(self) -> dict:
        n = len(self.samples)
        if n == 0:
            return {"n": 0}
        chars = sum(s.chars for s in self.samples)
        words = sum(s.words for s in self.samples)
        rates = [s.cer for s in self.samples]
        return {
            "n": n,
            "cer": sum(s.char_edits for s in self.samples) / max(chars, 1),
            "wer": sum(s.word_edits for s in self.samples) / max(words, 1),
            "median_cer": statistics.median(rates),
            "catastrophic": sum(r > CATASTROPHIC for r in rates) / n,
            "exact": sum(s.char_edits == 0 for s in self.samples) / n,
            "wrong_script": sum(s.wrong_script for s in self.samples) / n,
        }


def score(refs: Iterable[str], hyps: Iterable[str]) -> dict:
    sc = Scores()
    for r, h in zip(refs, hyps, strict=True):
        sc.add(r, h)
    return sc.summary()
