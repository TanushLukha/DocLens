"""Reading a page with M2's CRNN: the lines a finder gives, cut out upright, read, joined (M3).

A line much longer than its height would be squeezed: the recognizer's widest input is 24 times
its height, as in training (crnn.CRNNConfig.max_aspect). Such a line is cut at the gap between
words nearest its middle, again if need be, and the pieces' readings are joined with a space.
"""
from __future__ import annotations

import cv2
import numpy as np

from .layout import _runs
from .pages import crop_line
from .recognize import Recognizer


def pieces(crop: np.ndarray, max_aspect: float) -> list[np.ndarray]:
    """The crop, or its parts cut at word gaps, each at most max_aspect times as wide as high."""
    h, w = crop.shape[:2]
    if w <= max_aspect * h:
        return [crop]
    _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gaps = [(a + b) / 2 for a, b in _runs(~ink.any(axis=0)) if b - a >= 0.15 * h and 0 < a and b < w]
    if not gaps:
        return [crop]
    cut = int(min(gaps, key=lambda x: abs(x - w / 2)))
    if cut < h or cut > w - h:
        return [crop]
    return pieces(crop[:, :cut], max_aspect) + pieces(crop[:, cut:], max_aspect)


def read_page(rec: Recognizer, page: np.ndarray, quads: list[np.ndarray], margin: float = 0.15) -> list[str]:
    """The text of every line, in the order given."""
    parts, owner = [], []
    for i, q in enumerate(quads):
        for piece in pieces(crop_line(page, q, margin), rec.max_width / rec.height):
            parts.append(piece)
            owner.append(i)
    texts = rec.read(parts) if parts else []
    lines: list[list[str]] = [[] for _ in quads]
    for i, t in zip(owner, texts, strict=True):
        if t:
            lines[i].append(t)
    return [" ".join(t) for t in lines]
