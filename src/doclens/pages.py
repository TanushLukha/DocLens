"""Whole pages and the lines on them: where each line sits, what it says, and in which order.

A page set is a folder holding images/ and labels.json. For every page the file lists its image,
its lines in reading order (each a quadrilateral of four corners in page pixels: top left, top
right, bottom right, bottom left) with their text, and how the page was made. Synthetic pages come
from scripts/make_pages.py.

CORD's receipts become pages here. CORD marks every row of a receipt's item list, but not the
shop's details (blurred for privacy) and not every separator line, so each receipt is masked to
the part CORD annotates before any reader sees it (decision 29 in docs/PLAN.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import DATA
from .datasets import _cord_rows, _decode, _quad
from .text import normalize

PAGES = DATA / "pages"


@dataclass
class Line:
    quad: np.ndarray             # 4 x 2 float32: top left, top right, bottom right, bottom left
    text: str = ""


@dataclass
class Page:
    id: str
    lines: list[Line]            # in reading order
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    array: np.ndarray | None = None

    def image(self) -> np.ndarray:
        """Grayscale uint8."""
        if self.array is not None:
            return self.array
        img = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f"cannot read {self.path}")
        return img

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# geometry
def order_quad(pts) -> np.ndarray:
    """Four corners as top left, top right, bottom right, bottom left (image coordinates, y down).
    More than four points are first reduced to their smallest enclosing rectangle."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]   # clockwise on screen, from the top left side
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def box_quad(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def quad_size(quad: np.ndarray) -> tuple[float, float]:
    """(width, height) of a quadrilateral, along its own sides."""
    q = order_quad(quad)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    return float(w), float(h)


def crop_line(img: np.ndarray, quad: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """A line cut out upright: its quadrilateral, widened by `margin` times its height on every side,
    warped to a rectangle. The margin is real page around the line, as in the CORD crops M2 trained
    on; a tilted line comes out level, which M2's recognizer needs (skew was its weakest damage)."""
    q = order_quad(quad)
    w, h = quad_size(q)
    if w < 2 or h < 2:
        return np.full((8, 8), 255, np.uint8)
    m = margin * h
    u = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-6)       # along the line
    v = (q[3] - q[0]) / max(np.linalg.norm(q[3] - q[0]), 1e-6)       # down across it
    src = np.float32([q[0] - m * u - m * v, q[1] + m * u - m * v, q[2] + m * u + m * v, q[3] - m * u + m * v])
    W, H = int(round(w + 2 * m)), int(round(h + 2 * m))
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    t = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, t, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# page sets on disk
def write_set(folder: Path, pages: list[tuple[np.ndarray, Page]]) -> None:
    (folder / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for img, page in pages:
        file = f"images/{page.id}.png"
        if not cv2.imwrite(str(folder / file), img):
            raise OSError(f"could not write {folder / file}")
        rows.append({"id": page.id, "file": file, "meta": page.meta,
                     "lines": [{"quad": np.round(line.quad, 1).tolist(), "text": line.text} for line in page.lines]})
    (folder / "labels.json").write_text(json.dumps({"pages": rows}, ensure_ascii=False), encoding="utf8")


def read_set(name: str, root: Path | None = None) -> list[Page]:
    folder = (root or PAGES) / name
    rows = json.loads((folder / "labels.json").read_text(encoding="utf8"))["pages"]
    return [Page(r["id"], [Line(np.float32(line["quad"]), line["text"]) for line in r["lines"]], r["meta"],
                 folder / r["file"]) for r in rows]


# CORD receipts as pages
def _quads(entries) -> list[np.ndarray]:
    """CORD's lists of separators or blurred regions: entries of quads, or of words holding a quad."""
    out = []
    for entry in entries or []:
        for item in entry if isinstance(entry, list) else [entry]:
            out.append(_quad(item["quad"] if "quad" in item else item))
    return out


def mask_to_annotations(img: np.ndarray, words: list[np.ndarray], ignore: list[np.ndarray]) -> np.ndarray:
    """Paper over everything outside the annotated part (the hull of CORD's words, widened by one
    word height), and over separators and regions CORD marks as not to be read."""
    heights = [quad_size(q)[1] for q in words]
    pad = int(round(max(4.0, float(np.median(heights)))))
    keep = np.zeros(img.shape[:2], np.uint8)
    hull = cv2.convexHull(np.concatenate(words).astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(keep, hull, 255)
    keep = cv2.dilate(keep, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    paper = int(np.percentile(img[keep > 0], 80))
    out = img.copy()
    out[keep == 0] = paper
    for q in ignore:
        c = q.mean(axis=0)
        grown = c + (q - c) * 1.15 + np.sign(q - c) * 2           # a little beyond the quad
        cv2.fillConvexPoly(out, cv2.convexHull(grown.astype(np.float32)).astype(np.int32), paper)
    return out


def cord_pages(split: str = "test", root: Path | None = None) -> list[Page]:
    """CORD receipts as pages: one line per row (CORD's row_id), its words left to right, the rows
    top to bottom; each row's quadrilateral is the smallest rectangle around its words."""
    root = root or DATA / "cord"
    pages = []
    for ri, row in enumerate(_cord_rows(split, root)):
        img = _decode(row["image"])
        gt = json.loads(row["ground_truth"])
        words = [w for line in gt.get("valid_line", []) for w in line.get("words", [])]
        rows: dict[int, list] = {}
        for w in words:
            rows.setdefault(w.get("row_id", -1), []).append(w)
        lines = []
        for ws in rows.values():
            ws = sorted(ws, key=lambda w: float(_quad(w["quad"])[:, 0].min()))
            text = normalize(" ".join(w.get("text", "") for w in ws))
            if text:
                lines.append(Line(order_quad(np.concatenate([_quad(w["quad"]) for w in ws])), text))
        if not lines:
            continue
        lines.sort(key=lambda line: float(line.quad[:, 1].mean()))
        ignore = _quads(gt.get("repeating_symbol")) + _quads(gt.get("dontcare"))
        masked = mask_to_annotations(img, [_quad(w["quad"]) for w in words], ignore)
        pages.append(Page(str(ri), lines, {"set": f"cord_{split}", "script": "latn", "condition": "photo"},
                          array=masked))
    return pages
