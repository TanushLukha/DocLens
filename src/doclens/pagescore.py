"""Scoring whole pages: which lines a finder found, in which order, and what the page says.

* Lines. A found line matches a true line when their quadrilaterals overlap with an IoU of 0.5
  or more, each line used at most once (greedily, highest IoU first, as in ICDAR's text detection
  protocol). Precision is the share of found lines that match, recall the share of true lines.
* Splits and merges explain the misses: a true line is split when two or more found lines lie
  mostly (over half their area) inside it; a found line is a merge when it covers most of two or
  more true lines.
* Reading order: Kendall's tau between the true and the found order of the matched lines; 1 is
  the true order, minus 1 the reverse.
* Page CER: edits between the page's true text and the text read from it, over the length of the
  true text. Both pass through `normalize`, which turns every run of spaces and line breaks into
  one space, so a line cut in two costs nothing as long as its pieces are read in order. Runs of
  three or more of the same separator character (------, ======) are removed from both sides:
  CORD marks some of them and not others, so they are not counted for anyone.
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz.distance import Levenshtein
from shapely.geometry import Polygon

from .pages import Line, order_quad
from .text import normalize

SEPARATORS = re.compile(r"([-=_.*~#:+])\1{2,}")


def polygon(quad: np.ndarray) -> Polygon:
    p = Polygon(order_quad(quad))
    return p if p.is_valid else p.buffer(0)


def iou_matrix(true: list[np.ndarray], found: list[np.ndarray]) -> np.ndarray:
    tp, fp = [polygon(q) for q in true], [polygon(q) for q in found]
    out = np.zeros((len(tp), len(fp)))
    if not tp or not fp:
        return out
    tb = np.array([p.bounds for p in tp])
    fb = np.array([p.bounds for p in fp])
    for i, a in enumerate(tp):
        near = np.nonzero((fb[:, 0] < tb[i, 2]) & (fb[:, 2] > tb[i, 0]) & (fb[:, 1] < tb[i, 3]) & (fb[:, 3] > tb[i, 1]))[0]
        for j in near:
            inter = a.intersection(fp[j]).area
            if inter > 0:
                out[i, j] = inter / (a.area + fp[j].area - inter)
    return out


def match(iou: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """(true index, found index) pairs, one to one, highest IoU first."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    ii, jj = np.nonzero(iou >= threshold)
    used_t, used_f = set(), set()
    for k in np.argsort(-iou[ii, jj], kind="stable"):
        i, j = int(ii[k]), int(jj[k])
        if i not in used_t and j not in used_f:
            used_t.add(i)
            used_f.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def splits_merges(true: list[np.ndarray], found: list[np.ndarray], share: float = 0.5) -> tuple[int, int]:
    tp, fp = [polygon(q) for q in true], [polygon(q) for q in found]
    inside = np.zeros((len(tp), len(fp)))          # inter / found area
    covers = np.zeros((len(tp), len(fp)))          # inter / true area
    for i, a in enumerate(tp):
        for j, b in enumerate(fp):
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a.bounds, b.bounds
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                inter = a.intersection(b).area
                inside[i, j] = inter / max(b.area, 1e-9)
                covers[i, j] = inter / max(a.area, 1e-9)
    splits = int(((inside > share).sum(axis=1) >= 2).sum()) if fp else 0
    merges = int(((covers > share).sum(axis=0) >= 2).sum()) if tp else 0
    return splits, merges


def kendall(pairs: list[tuple[int, int]]) -> float | None:
    """Kendall's tau between true and found positions of matched lines (None with fewer than two)."""
    if len(pairs) < 2:
        return None
    found = [j for _, j in sorted(pairs)]
    n = len(found)
    s = sum(np.sign(np.array(found[k + 1:]) - found[k]).sum() for k in range(n - 1))
    return float(s) / (n * (n - 1) / 2)


def page_text(text: str) -> str:
    return " ".join(SEPARATORS.sub(" ", normalize(text)).split())


def page_edits(ref: str, hyp: str) -> tuple[int, int]:
    r, h = page_text(ref), page_text(hyp)
    return Levenshtein.distance(r, h), len(r)


def score_page(true: list[Line], found: list[np.ndarray] | None, text: str) -> dict:
    """Everything for one page and one reader. `found` is None for a reader whose lines are not
    measured (the true lines themselves)."""
    edits, chars = page_edits("\n".join(line.text for line in true), text)
    out: dict = {"edits": edits, "chars": chars, "true_lines": len(true)}
    if found is not None:
        tq = [line.quad for line in true]
        pairs = match(iou_matrix(tq, found))
        splits, merges = splits_merges(tq, found)
        out.update(found_lines=len(found), matched=len(pairs), splits=splits, merges=merges, tau=kendall(pairs))
    return out
