"""Synthetic pages: held out text set in one or two columns with known lines, then damaged.

A page is A4 at 300 dpi (2480 x 3508 pixels), the usual resolution for scanning books: a title,
one or two columns of paragraphs and a page number. Text is wrapped by its drawn width; every line
is drawn by render_line, so Devanagari is shaped exactly as in M1 and M2, and lines sit on a
regular baseline grid. A line's truth is the tightest box around its ink, kept as a quadrilateral
through the damage: the rotation or perspective that moves the pixels moves its corners too.

Damage (CONDITIONS) follows M1 at page scale: clean; scanner noise; a shadow across the page; the
whole page turned by 1 to 4 degrees; a phone photo (perspective, uneven light, soft focus, JPEG).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from . import degrade
from .fonts import Font
from .pages import Line, box_quad
from .render import render_line, text_width

A4 = (2480, 3508)                                   # width, height at 300 dpi
CONDITIONS = ("clean", "noise", "shadow", "skew", "photo")


@dataclass
class Style:
    size: int                    # body font size in pixels (40 to 52 is 10 to 12.5 points at 300 dpi)
    pitch: int                   # baseline to baseline
    margins: tuple[int, int, int, int]   # left, top, right, bottom
    columns: int
    gap: int                     # between columns
    para_gap: int                # extra space between paragraphs
    indent: int                  # first line of a paragraph
    title_size: int
    title_centred: bool
    ink: int
    paper: int
    latin_in_deva_font: bool


def draw_style(rng: np.random.Generator, columns: int) -> Style:
    size = int(rng.integers(40, 53))
    return Style(size=size, pitch=int(size * rng.uniform(1.4, 1.75)),
                 margins=(int(rng.integers(180, 280)), int(rng.integers(200, 300)),
                          int(rng.integers(180, 280)), int(rng.integers(240, 340))),
                 columns=columns, gap=int(rng.integers(90, 150)), para_gap=int(size * rng.uniform(0.3, 0.9)),
                 indent=int(size * rng.choice([0.0, 1.5, 2.5])), title_size=int(size * rng.uniform(1.5, 1.9)),
                 title_centred=bool(rng.random() < 0.6), ink=int(rng.integers(0, 50)),
                 paper=int(rng.integers(225, 256)), latin_in_deva_font=bool(rng.random() < 0.5))


def wrap(words: list[str], width: float, measure: Callable[[str], float], indent: float = 0.0) -> list[str]:
    """Greedy word wrap by drawn width; a word wider than a whole line gets a line of its own."""
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        room = width - (indent if not lines else 0.0)
        if cur and measure(" ".join(cur + [w])) > room:
            lines.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


class Canvas:
    """A page being set: lines are drawn onto it and their ink boxes recorded."""

    def __init__(self, style: Style, deva: Font, latn: Font):
        self.s, self.deva, self.latn = style, deva, latn
        self.img = np.full((A4[1], A4[0]), style.paper, np.uint8)
        self.lines: list[Line] = []

    def measure(self, size: int) -> Callable[[str], float]:
        return lambda t: text_width(t, self.deva, self.latn, size, self.s.latin_in_deva_font)

    def drawable(self, text: str) -> list[str]:
        """The words of text that the page's two fonts can draw (Wikipedia text has a few names in
        other scripts); the truth holds only what is drawn."""
        out = []
        for w in text.split():
            try:
                text_width(w, self.deva, self.latn, 10, self.s.latin_in_deva_font)
                out.append(w)
            except ValueError:
                pass
        return out

    def put(self, text: str, x: float, baseline: float, size: int) -> bool:
        """Draw text with its left end at x and its baseline at `baseline`; False if a character
        has no font or the line would leave the page."""
        try:
            r = render_line(text, self.deva, self.latn, size, latin_in_deva_font=self.s.latin_in_deva_font,
                            ink=self.s.ink, paper=self.s.paper, pad=(4, 4, 4, 4))
        except ValueError:
            return False
        top, left = int(round(baseline)) - r.baseline, int(round(x)) - 4
        h, w = r.image.shape
        if top < 0 or left < 0 or top + h > self.img.shape[0] or left + w > self.img.shape[1]:
            return False
        ys, xs = np.nonzero(r.image < (self.s.ink + self.s.paper) / 2)
        if len(xs) == 0:
            return False
        region = self.img[top:top + h, left:left + w]
        np.minimum(region, r.image, out=region)
        self.lines.append(Line(box_quad(left + xs.min(), top + ys.min(), left + xs.max() + 1, top + ys.max() + 1), text))
        return True


def compose(title: str, paragraphs: Iterator[str], page_no: int, deva: Font, latn: Font,
            style: Style) -> tuple[np.ndarray, list[Line]]:
    """Set a page: the title across the text width, paragraphs flowing down the columns until the
    page is full, the page number at the bottom. Lines come back in reading order."""
    s = style
    c = Canvas(s, deva, latn)
    left, top, right, bottom = s.margins
    text_w = A4[0] - left - right
    baseline = top + s.title_size
    for t in wrap(c.drawable(title), text_w, c.measure(s.title_size)):
        w = c.measure(s.title_size)(t)
        x = left + (text_w - w) / 2 if s.title_centred else left
        if c.put(t, x, baseline, s.title_size):
            baseline += int(s.title_size * 1.3)
    body_top = baseline + s.pitch
    col_w = (text_w - s.gap * (s.columns - 1)) / s.columns
    floor = A4[1] - bottom - s.size                 # no baseline below this
    col, baseline = 0, body_top
    measure = c.measure(s.size)
    full = False
    for para in paragraphs:
        x0 = left + col * (col_w + s.gap)
        for i, t in enumerate(wrap(c.drawable(para), col_w, measure, s.indent)):
            if baseline > floor:
                col += 1
                if col == s.columns:
                    full = True
                    break
                baseline, x0 = body_top, left + col * (col_w + s.gap)
            if c.put(t, x0 + (s.indent if i == 0 else 0), baseline, s.size):
                baseline += s.pitch
        if full:
            break
        baseline += s.para_gap
    number = str(page_no)
    c.put(number, left + (text_w - measure(number)) / 2, A4[1] - bottom + 2 * s.size, s.size)
    return c.img, c.lines


# damage that moves pixels moves the line corners the same way
def _transform(lines: list[Line], m: np.ndarray) -> list[Line]:
    out = []
    for line in lines:
        pts = line.quad.reshape(-1, 1, 2).astype(np.float32)
        q = cv2.perspectiveTransform(pts, m) if m.shape == (3, 3) else cv2.transform(pts, m)
        out.append(Line(q.reshape(4, 2).astype(np.float32), line.text))
    return out


def rotate(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn the page about its centre, growing the canvas so no corner is cut (as degrade.rotate)."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=int(np.median(img))), m


def perspective(img: np.ndarray, rng: np.random.Generator, strength: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-strength, strength, (4, 2)) * np.float32([w * 0.25, h])).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(np.median(img))), m


def damage(img: np.ndarray, lines: list[Line], condition: str,
           rng: np.random.Generator) -> tuple[np.ndarray, list[Line]]:
    if condition == "clean":
        return img, lines
    if condition == "noise":
        return degrade.noise(img, rng), lines
    if condition == "shadow":
        return degrade.shadow(img, rng), lines
    if condition == "skew":
        img, m = rotate(img, float(rng.uniform(1.0, 4.0)) * (1 if rng.random() < 0.5 else -1))
        return img, _transform(lines, m)
    if condition == "photo":
        img, m = perspective(img, rng)
        img = degrade.shadow(img, rng, (0.25, 0.45))
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.6, 1.0)))
        return degrade.jpeg(img, rng, (35, 55)), _transform(lines, m)
    raise ValueError(f"unknown condition {condition}")
