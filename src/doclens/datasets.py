"""The evaluation sets, read into one shape: an image, its true text, and where it came from.

* Mozhi Hindi (IIIT Hyderabad, CC BY 4.0): 100,049 printed Hindi words cropped from 1,000
  book pages scanned at 600 DPI, each transcribed by hand. Real ink, real paper. The crops
  are 48 pixels high and end at the ink, so the reader adds a margin (see `add_margin`).
* CORD v2 (Clova AI, CC BY 4.0): 1,000 photographed receipts with a box and the text of
  every word; used here for real English words and lines, and in M4 for fields.
* Synthetic lines (made by scripts/make_synth.py): held out text, rendered with known
  fonts, then damaged in known ways, so every condition can be measured on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import DATA
from .text import normalize, script_of


@dataclass
class Item:
    text: str
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    array: np.ndarray | None = None
    margin: float = 0.0           # border added around the image, as a fraction of its height

    def image(self) -> np.ndarray:
        """Grayscale uint8."""
        if self.array is not None:
            img = self.array
        else:
            img = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"cannot read {self.path}")
        return add_margin(img, self.margin) if self.margin else img


def add_margin(img: np.ndarray, frac: float) -> np.ndarray:
    """A border of paper colour (the image's 90th percentile), frac times the height, on every side.

    Recognizers need some background around text. In 98% of Mozhi crops the ink comes within
    two pixels of an edge, and Tesseract misreads many more words that way (M1's margins
    experiment measures how many). CORD crops keep 15% of the real receipt around each word
    instead; synthetic lines are drawn with padding.
    """
    b = int(round(frac * img.shape[0]))
    if b <= 0:
        return img
    paper = int(np.percentile(img, 90))
    return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=paper)


# Mozhi
def _gt_file(folder: Path) -> Path:
    cands = sorted(folder.rglob("*gt*.txt")) or sorted(p for p in folder.rglob("*.txt") if "vocab" not in p.name)
    if not cands:
        raise FileNotFoundError(f"no ground truth file under {folder}")
    return cands[0]


MOZHI_MARGIN = 0.25


def mozhi(split: str = "test", root: Path | None = None, limit: int | None = None,
          margin: float = MOZHI_MARGIN) -> list[Item]:
    """Mozhi Hindi word images. Expects data/mozhi/hindi/<split>/ as unzipped by get_data.py."""
    folder = (root or DATA / "mozhi" / "hindi") / split
    gt = _gt_file(folder)
    items = []
    with open(gt, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
            if len(parts) != 2:
                continue
            rel, text = parts
            path = _resolve(folder, gt.parent, rel)
            items.append(Item(normalize(text), {"set": f"mozhi_{split}", "id": rel, "script": "deva"}, path,
                              margin=margin))
            if limit and len(items) >= limit:
                break
    if not items:
        raise ValueError(f"no image and text pairs found in {gt}")
    return items


def _resolve(folder: Path, base: Path, rel: str) -> Path:
    for cand in (base / rel, folder / rel, folder / "images" / rel, base / "images" / Path(rel).name):
        if cand.exists():
            return cand
    return base / rel


# CORD
def _cord_rows(split: str, root: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq
    files = sorted((root / "data").glob(f"{split}-*.parquet")) or sorted(root.rglob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {root}")
    for f in files:
        table = pq.read_table(f)
        for row in table.to_pylist():
            yield row


def _decode(image_field) -> np.ndarray:
    data = image_field["bytes"] if isinstance(image_field, dict) else image_field
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("cannot decode a CORD image")
    return img


def _quad(q) -> np.ndarray:
    """CORD writes a quad as {x1, y1, ..., x4, y4}, clockwise from the top left; lists are accepted too."""
    if isinstance(q, dict):
        return np.float32([[q["x1"], q["y1"]], [q["x2"], q["y2"]], [q["x3"], q["y3"]], [q["x4"], q["y4"]]])
    return np.float32(q).reshape(4, 2)


def crop_quad(img: np.ndarray, quad: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """Warp a (possibly tilted) word quadrilateral to an upright rectangle, with a margin."""
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    if w < 2 or h < 2:
        raise ValueError("degenerate quad")
    m = int(round(margin * h))
    dst = np.float32([[m, m], [m + w, m], [m + w, m + h], [m, m + h]])
    t = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, t, (w + 2 * m, h + 2 * m), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def cord(split: str = "test", level: str = "word", root: Path | None = None) -> list[Item]:
    """CORD v2 words or lines. A line is the words of one receipt that share a row_id, left to right."""
    root = root or DATA / "cord"
    items = []
    for ri, row in enumerate(_cord_rows(split, root)):
        img = _decode(row["image"])
        gt = json.loads(row["ground_truth"])
        words = [w for line in gt.get("valid_line", []) for w in line.get("words", [])]
        if level == "word":
            for wi, w in enumerate(words):
                text = normalize(w.get("text", ""))
                if not text:
                    continue
                try:
                    crop = crop_quad(img, _quad(w["quad"]))
                except ValueError:
                    continue
                items.append(Item(text, {"set": f"cord_{split}_word", "id": f"{ri}_{wi}", "script": script_of(text)},
                                  array=crop))
        else:
            rows: dict[int, list] = {}
            for w in words:
                rows.setdefault(w.get("row_id", -1), []).append(w)
            for rid, ws in rows.items():
                ws.sort(key=lambda w: float(_quad(w["quad"])[:, 0].min()))
                text = normalize(" ".join(w.get("text", "") for w in ws))
                if not text:
                    continue
                pts = np.concatenate([_quad(w["quad"]) for w in ws])
                x0, y0 = np.floor(pts.min(axis=0)).astype(int)
                x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
                pad = max(2, int(0.15 * (y1 - y0)))
                x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
                crop = img[y0:y1 + pad, x0:x1 + pad]
                if crop.size == 0:
                    continue
                items.append(Item(text, {"set": f"cord_{split}_line", "id": f"{ri}_{rid}", "script": script_of(text)},
                                  array=crop))
    return items


# synthetic lines
FIELDS = ["file", "text", "condition", "fonts", "script", "size"]


def read_tsv(path: Path) -> list[dict]:
    """Plain tab separated rows with a header line: no quoting, so any character but tab and newline is safe."""
    with open(path, encoding="utf8") as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    out = ["\t".join(header)]
    for r in rows:
        cells = [str(c) for c in r]
        if any("\t" in c or "\n" in c for c in cells):
            raise ValueError(f"a cell contains a tab or a line break: {cells}")
        out.append("\t".join(cells))
    Path(path).write_text("\n".join(out) + "\n", encoding="utf8")


def synth(name: str, root: Path | None = None) -> list[Item]:
    """A set written by make_synth.py: images/ plus labels.tsv (file, text, condition, fonts, script, size)."""
    folder = (root or DATA / "synth") / name
    items = []
    for r in read_tsv(folder / "labels.tsv"):
        meta = {"set": name, "id": r["file"], "condition": r["condition"], "fonts": r["fonts"],
                "script": r["script"], "size": int(r["size"])}
        items.append(Item(r["text"], meta, folder / r["file"]))
    return items


def write_synth(folder: Path, rows: list[dict]) -> None:
    write_tsv(folder / "labels.tsv", FIELDS, [[r[k] for k in FIELDS] for r in rows])
