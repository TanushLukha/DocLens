"""A classical line finder: OpenCV and projection profiles, nothing learned (M3).

1. Ink. The page is scaled to at most 1,600 pixels tall (body text becomes about 15 to 25 pixels
   high: enough to find lines, and quick) and binarised with Sauvola's local threshold, which M1
   found best under shadows.
2. Angle. The rotation that makes the rows of ink sharpest (preprocess.skew_angle, M1's deskewing)
   turns the ink level; every result is turned back at the end.
3. Components. Connected components much smaller or much taller than the median text height are
   dropped as specks and frames, and short flat ones as the dashes of separator lines (------,
   =====, which would bridge columns); the rest are the units everything below works with.
4. Blocks in reading order: a recursive XY cut over the components. Columns first: a region is
   cut down at the widest gap between columns, searched also at slight slants, since a
   photographed page tilts its columns by different angles. Only tall regions are cut down (a
   title's gaps are between words), and only between sides of about the same width with a gap much
   narrower than they are: a receipt's prices belong to the item names beside them. When no cut down is
   possible, the region is cut across: below a title or above a page number that reaches into the
   gap between columns (a strip of at most a few lines), else preferably where the cut frees
   columns; without columns, top to bottom is the reading order anyway.
5. Lines. Each block is levelled again by its own angle (what is left of a page's turn there). The
   ink per row rises on every line and falls between lines; rows holding a good share of the ink of
   the strongest row nearby are the lines' cores, and every component joins the core it overlaps
   most, or the nearest one (a vowel sign above or below a line follows the line it sits on). A line
   is the box around its components, turned back to the page.

Distances are multiples of the median text height or of the line pitch, so one setting serves book
pages and receipts; they were chosen on the validation pages and receipts, never on a test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pages import box_quad
from .preprocess import sauvola, skew_angle


@dataclass(frozen=True)
class Params:
    work_height: int = 1600       # pages taller than this are scaled down for the analysis
    window: float = 3.0           # Sauvola window, in text heights
    k: float = 0.1                # Sauvola's k (low: faded thermal print is light grey)
    close: int = 3                # ink is closed with a square this many pixels wide before finding components
    speck: float = 0.03           # components with less area than this times h squared are specks
    tall: float = 5.0             # components taller than this many text heights are not text
    flat: float = 0.4             # short components flatter than this many text heights are dashes, not letters
    col_gap: float = 0.8          # a gap between columns is at least this many line pitches wide
    col_rows: float = 3.0         # and its region at least this many line pitches tall
    slant: float = 0.08           # gaps between columns are also searched at slants up to this (dx per dy)
    balance: float = 0.75         # sides at least this share of each other's width,
    wide: float = 3.0             # and this many times as wide as the gap between them, are columns
    row_gap: float = 0.9          # an empty band across a region must be this many text heights tall to cut it
    strip: float = 4.0            # a title or page number cut off above or below columns is at most this many
                                  # line pitches tall: two title lines at the largest title size are 3.4
    local_angle: float = 4.0      # each block is levelled again, up to this many degrees
    core: float = 0.2             # rows with more ink than this share of the strongest row within half a pitch
                                  # are a line's core
    min_core: float = 0.3         # cores thinner than this many text heights join their neighbour
    tall_core: float = 1.8        # cores taller than this many typical cores hold two touching lines


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist(), strict=True))


def _covered(a: np.ndarray, b: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Which positions of [lo, hi) fall inside at least one interval [a, b)."""
    d = np.zeros(hi - lo + 1, np.int64)
    np.add.at(d, np.clip(np.floor(a).astype(np.int64) - lo, 0, hi - lo), 1)
    np.add.at(d, np.clip(np.ceil(b).astype(np.int64) - lo, 0, hi - lo), -1)
    return np.cumsum(d)[:-1] > 0


def _turn(pts: np.ndarray, angle: float, centre: np.ndarray) -> np.ndarray:
    """Points turned as cv2.getRotationMatrix2D(centre, angle) turns an image (degrees)."""
    if abs(angle) < 1e-6:
        return pts
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle, 1.0)
    return pts @ m[:, :2].T + m[:, 2]


def _rotate(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0), m


class LineFinder:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()

    def __call__(self, page: np.ndarray) -> list[np.ndarray]:
        """Line quadrilaterals (4 x 2, page pixels) in reading order."""
        p = self.p
        gray = page if page.ndim == 2 else cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        s = min(1.0, p.work_height / gray.shape[0])
        work = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        angle = skew_angle(work, max_angle=8.0)
        ink = (sauvola(work, window=self._window(work), k=p.k) == 0).astype(np.uint8)
        if p.close > 1:                 # join the dots of dot matrix print, and broken strokes of faded print
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((p.close, p.close), np.uint8))
        ink, m = _rotate(ink, angle)
        if not self._components(ink):
            return []
        inv = cv2.invertAffineTransform(m)
        quads = []
        for block in self._xy_cut(np.arange(len(self.boxes))):
            for q in self._lines(block):
                quads.append(cv2.transform(q.reshape(-1, 1, 2), inv).reshape(4, 2) / s)
        return [q.astype(np.float32) for q in quads]

    def _window(self, work: np.ndarray) -> int:
        """Sauvola's window from a first guess of the text height (Otsu components)."""
        _, bw = cv2.threshold(work, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        hs = st[1:, cv2.CC_STAT_HEIGHT]
        h = float(np.median(hs[hs >= 4])) if (hs >= 4).any() else 20.0
        return int(max(15, round(self.p.window * h)) // 2 * 2 + 1)

    def _components(self, ink: np.ndarray) -> bool:
        """Keeps the text components: their boxes, label ids and image, the median text height and
        the line pitch. False when the page has no ink."""
        _, labels, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        hs, ws, areas = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_AREA]
        big = (hs >= 4) & (areas >= 8)
        if not big.any():
            return False
        h = float(np.median(hs[big]))
        dash = (hs < self.p.flat * h) & (ws < 3 * h)           # dashes and dots of a separator line
        keep = (areas >= self.p.speck * h * h) & (hs <= self.p.tall * h) & (ws <= 60 * h) & ~dash
        if not keep.any():
            return False
        x, y = st[1:, cv2.CC_STAT_LEFT], st[1:, cv2.CC_STAT_TOP]
        self.boxes = np.stack([x, y, x + ws, y + hs], axis=1)[keep].astype(np.float64)
        self.ids = np.nonzero(keep)[0] + 1
        self.labels, self.h = labels, h
        rows = _covered(self.boxes[:, 1], self.boxes[:, 3], 0, labels.shape[0])
        steps = np.diff([a for a, _ in _runs(rows)])
        steps = steps[(steps > 0.5 * h) & (steps < 4 * h)]
        self.pitch = float(np.median(steps)) if len(steps) else 1.5 * h
        return True

    # blocks
    def _xy_cut(self, idx: np.ndarray) -> list[np.ndarray]:
        if len(idx) == 0:
            return []
        down = self._down(idx)
        if down is not None:
            return self._xy_cut(down[0]) + self._xy_cut(down[1])
        across = self._across_cuts(idx)
        if not across:
            return [idx]
        # columns hidden by a title above them or a page number below: cut off the strip that blocks the
        # gap. Only a short strip can be one; taken for a title, the top half of two columns would be
        # read before their bottom half.
        by_place = sorted(across, key=lambda tb: self.boxes[tb[0], 3].max())
        if len(by_place) > 1:
            (head, after_head), (before_foot, foot) = by_place[0], by_place[-1]
            head, foot = (s if self._short(s) else s[:0] for s in (head, foot))
            if len(head) or len(foot):
                gap = self._down(np.setdiff1d(idx, np.concatenate([head, foot])))
                if gap is not None:
                    if len(head) and self._blocks(head, gap):
                        return self._xy_cut(head) + self._xy_cut(after_head)
                    if len(foot) and self._blocks(foot, gap):
                        return self._xy_cut(before_foot) + self._xy_cut(foot)
        top, bottom = next(((t, b) for t, b in across[:8] if self._down(t) is not None or self._down(b) is not None),
                           across[0])
        return self._xy_cut(top) + self._xy_cut(bottom)

    def _short(self, idx: np.ndarray) -> bool:
        """Whether these components fit in a strip short enough to be a title or a page number."""
        b = self.boxes[idx]
        return bool(b[:, 3].max() - b[:, 1].min() <= self.p.strip * self.pitch)

    def _across_cuts(self, idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every empty band across the region tall enough to cut at, widest first, as (top, bottom)."""
        b = self.boxes[idx]
        lo, hi = int(np.floor(b[:, 1].min())), int(np.ceil(b[:, 3].max()))
        gaps = [(e - a, lo + (a + e) / 2) for a, e in _runs(~_covered(b[:, 1], b[:, 3], lo, hi))
                if e - a >= self.p.row_gap * self.h]
        mid = (b[:, 1] + b[:, 3]) / 2
        return [(idx[mid < y], idx[mid >= y]) for _, y in sorted(gaps, reverse=True)]

    def _down(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | None:
        """The widest gap between columns, at the slant that opens it most: (left, right, slant, gap
        start, gap end, centre row), positions measured after the slant; None without columns."""
        p = self.p
        if len(idx) < 4:
            return None
        b = self.boxes[idx]
        if b[:, 3].max() - b[:, 1].min() < p.col_rows * self.pitch:
            return None
        cy = (b[:, 1] + b[:, 3]) / 2
        ym = (cy.min() + cy.max()) / 2
        best, width = None, p.col_gap * self.pitch
        for slant in np.linspace(-p.slant, p.slant, 9):
            shift = -slant * (cy - ym)
            x0, x1 = b[:, 0] + shift, b[:, 2] + shift
            lo, hi = int(np.floor(x0.min())), int(np.ceil(x1.max()))
            for a, e in _runs(~_covered(x0, x1, lo, hi)):
                if e - a >= width:
                    left = (x0 + x1) / 2 < lo + (a + e) / 2
                    if left.any() and (~left).any() and self._may_cut_down(x0, x1, left, e - a):
                        best, width = (idx[left], idx[~left], float(slant), lo + a, lo + e, float(ym)), e - a
        return best

    def _blocks(self, idx: np.ndarray, gap: tuple) -> bool:
        """Whether any of these components reaches into the gap between columns."""
        _, _, slant, g0, g1, ym = gap
        b = self.boxes[idx]
        shift = -slant * ((b[:, 1] + b[:, 3]) / 2 - ym)
        return bool(((b[:, 0] + shift < g1) & (b[:, 2] + shift > g0)).any())

    def _may_cut_down(self, x0: np.ndarray, x1: np.ndarray, left: np.ndarray, gap: float) -> bool:
        """Columns are sides of about the same width with a gap much narrower than they are; a
        receipt's item names and prices are not."""
        wl, wr = x1[left].max() - x0[left].min(), x1[~left].max() - x0[~left].min()
        return bool(min(wl, wr) >= self.p.balance * max(wl, wr) and min(wl, wr) >= self.p.wide * gap)

    # lines
    def _lines(self, idx: np.ndarray) -> list[np.ndarray]:
        """The block's lines as quadrilaterals in the levelled page's coordinates."""
        p = self.p
        b = self.boxes[idx]
        x0, y0, x1, y1 = int(b[:, 0].min()), int(b[:, 1].min()), int(np.ceil(b[:, 2].max())), int(np.ceil(b[:, 3].max()))
        angle = 0.0
        if len(idx) > 3:
            ink = np.isin(self.labels[y0:y1, x0:x1], self.ids[idx])
            angle = skew_angle(np.where(ink, 0, 255).astype(np.uint8), max_angle=p.local_angle)
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)
        level = _turn(corners.reshape(-1, 2), angle, centre).reshape(-1, 4, 2)
        lb = np.concatenate([level.min(axis=1), level.max(axis=1)], axis=1)   # boxes in the block's level frame
        top = int(np.floor(lb[:, 1].min()))
        prof = np.zeros(int(np.ceil(lb[:, 3].max())) - top + 1)
        for bx0, by0, bx1, by1 in lb:                                          # ink per row, from widths
            prof[int(by0) - top:int(np.ceil(by1)) - top] += bx1 - bx0
        # a row is in a line's core when it holds a good share of the ink of the strongest row within half a
        # pitch: a paragraph's short last line counts as fully as a long one, a row of vowel signs does not
        k = max(3, int(self.pitch) // 2 * 2 + 1)
        local = cv2.dilate(prof.astype(np.float32).reshape(-1, 1), np.ones((k, 1), np.uint8)).ravel()
        cores = self._tidy(_runs(prof > p.core * local), prof)
        if not cores:
            return []
        cy0, cy1 = np.array([c0 for c0, _ in cores]) + top, np.array([c1 for _, c1 in cores]) + top
        groups: list[list[int]] = [[] for _ in cores]
        for k, (_, by0, _, by1) in enumerate(lb):
            overlap = np.minimum(cy1, by1) - np.maximum(cy0, by0)
            if overlap.max() > 0:
                groups[int(np.argmax(overlap))].append(k)
            else:
                mid = (by0 + by1) / 2
                groups[int(np.argmin(np.maximum(cy0 - mid, mid - cy1)))].append(k)
        out = []
        for g in groups:
            if g:
                gb = lb[g]
                q = box_quad(gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()).astype(np.float64)
                out.append(_turn(q, -angle, centre).astype(np.float32))
        return out

    def _tidy(self, cores: list[tuple[int, int]], prof: np.ndarray) -> list[tuple[int, int]]:
        """Thin cores join their nearest neighbour; cores much taller than usual are split at their
        weakest row (two lines touching)."""
        p, h = self.p, self.h
        out: list[tuple[int, int]] = []
        for s, e in cores:
            if out and (e - s < p.min_core * h or out[-1][1] - out[-1][0] < p.min_core * h) and s - out[-1][1] < 0.5 * h:
                out[-1] = (out[-1][0], e)
            else:
                out.append((s, e))
        if len(out) < 2:
            return out
        usual = float(np.median([e - s for s, e in out]))
        split: list[tuple[int, int]] = []
        for s, e in out:
            while e - s > p.tall_core * usual:
                lo, hi = s + int(0.3 * (e - s)), s + int(0.7 * (e - s))
                cut = lo + int(np.argmin(prof[lo:hi]))
                split.append((s, cut))
                s = cut
            split.append((s, e))
        return split
