"""Draws a line of text as a grayscale image, the way a printed document shows it.

Hindi documents usually set Devanagari in one font and English words, digits and
symbols in another (or in the same one, when it covers Latin). A line is therefore split
into runs, each run drawn with a font that has every one of its characters, and the
runs share one baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .fonts import Font
from .text import is_devanagari


@dataclass
class Rendered:
    image: np.ndarray            # uint8, H x W, dark text on light paper
    text: str
    fonts: tuple[str, ...]
    size: int


def runs(text: str, deva: Font, latn: Font, latin_in_deva_font: bool) -> list[tuple[str, Font]]:
    """Split text into (substring, font) runs. Spaces stay with the run they follow."""
    out: list[tuple[str, Font]] = []
    for ch in text:
        if ch == " " and out:
            font = out[-1][1]
        elif is_devanagari(ch) or (latin_in_deva_font and deva.covers(ch)):
            font = deva
        else:
            font = latn
        if not font.covers(ch):
            font = latn if latn.covers(ch) else deva
            if not font.covers(ch):
                raise ValueError(f"no font covers {ch!r} (U+{ord(ch):04X})")
        if out and out[-1][1] is font:
            out[-1] = (out[-1][0] + ch, font)
        else:
            out.append((ch, font))
    return out


def render_line(text: str, deva: Font, latn: Font, size: int, *, latin_in_deva_font: bool = True,
                ink: int = 20, paper: int = 245, pad: tuple[int, int, int, int] = (12, 10, 12, 10)) -> Rendered:
    """Render one line. pad is (left, top, right, bottom) in pixels around the ink."""
    parts = runs(text, deva, latn, latin_in_deva_font)
    x, above, below, boxes = 0.0, 0, 0, []
    for s, f in parts:
        pil = f.pil(size)
        left, top, right, bottom = pil.getbbox(s, anchor="ls")     # ink box relative to the baseline
        boxes.append((x, s, pil))
        above, below = max(above, -top), max(below, bottom)
        x += pil.getlength(s)
    w = int(np.ceil(x)) + pad[0] + pad[2]
    h = above + below + pad[1] + pad[3]
    img = Image.new("L", (max(w, 1), max(h, 1)), paper)
    draw = ImageDraw.Draw(img)
    for x0, s, pil in boxes:
        draw.text((pad[0] + x0, pad[1] + above), s, font=pil, fill=ink, anchor="ls")
    return Rendered(np.asarray(img), text, tuple(dict.fromkeys(f.name for _, f in parts)), size)
