"""Fields on receipts (M4): CORD's words with their categories, tags, and field scores.

CORD marks every annotated word with the category of its group: the name of an item (menu.nm),
its count and price, the subtotal, the tax, the total and so on. A field is one group: its
category and its words. A model tags every word with BIO tags (B for the first word of a field, I
for the words after it), and the fields come back from the tags.

Words are kept in reading order: CORD's rows top to bottom, words left to right in a row. OCR
words come in the same order (docTR's rows, M3), so a model reads true words and OCR words the
same way.

A field is right when its category and its text are both right: a total with one wrong digit is a
wrong total. Field F1 counts, per receipt, the (category, text) pairs found in both the truth and
the prediction, texts compared without spaces (decision 41 in docs/PLAN.md). CORD sometimes
makes a separator (":" or "-") a field of its own inside another field of the same category;
tags cannot say which of the two a later word belongs to, so even perfect tags miss those few
fields.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import DATA
from .datasets import _cord_rows, _decode, _quad
from .pages import _quads, mask_to_annotations
from .text import normalize


@dataclass
class Receipt:
    id: str
    words: list[str]                     # in reading order
    quads: list[np.ndarray]              # one 4 x 2 quadrilateral per word, receipt pixels
    size: tuple[int, int]                # width, height
    tags: list[str] = field(default_factory=list)       # BIO tag per word ("O" when unknown)
    groups: list[int] = field(default_factory=list)     # the field (group) each word belongs to
    image: np.ndarray | None = None      # the masked receipt (see iter_cord)

    def fields(self) -> list[tuple[str, str]]:
        """The true fields: (category, text) per group, words in reading order."""
        by: dict[int, list[int]] = {}
        for i, g in enumerate(self.groups):
            if g >= 0:
                by.setdefault(g, []).append(i)
        out = []
        for _, idx in sorted(by.items(), key=lambda kv: kv[1][0]):
            out.append((self.tags[idx[0]][2:], field_text(self.words[i] for i in idx)))
        return out


def field_text(words) -> str:
    return " ".join(normalize(" ".join(words)).split())


def reading_order(rows: list[int], quads: list[np.ndarray]) -> list[int]:
    """Word indices in reading order: rows (one id per word) top to bottom by their mean height,
    words left to right within a row."""
    by: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r, []).append(i)
    ordered = sorted(by.values(), key=lambda idx: float(np.mean([quads[i][:, 1].mean() for i in idx])))
    return [i for idx in ordered for i in sorted(idx, key=lambda i: float(quads[i][:, 0].min()))]


def bio(groups: list[int], categories: list[str]) -> list[str]:
    """B for the first word of each group in the order given, I for the rest; O without a group."""
    seen: set[int] = set()
    out = []
    for g, c in zip(groups, categories, strict=True):
        if g < 0:
            out.append("O")
        else:
            out.append(("I-" if g in seen else "B-") + c)
            seen.add(g)
    return out


def iter_cord(split: str = "test", root: Path | None = None, image: str = "small") -> Iterator[Receipt]:
    """CORD's receipts one by one, as words in reading order with their BIO tags. The image is
    masked to the annotated part, as in M3 (decision 29): image "full" keeps it at its size,
    "small" as the 224 x 224 RGB image LayoutLMv3 takes, "none" drops it."""
    root = root or DATA / "cord"
    for ri, row in enumerate(_cord_rows(split, root)):
        gt = json.loads(row["ground_truth"])
        words, quads, rows, groups, cats, every = [], [], [], [], [], []
        for gi, line in enumerate(gt.get("valid_line", [])):
            for w in line.get("words", []):
                q = _quad(w["quad"])
                every.append(q)
                text = normalize(w.get("text", "")).strip()
                if not text:
                    continue
                words.append(text)
                quads.append(q)
                rows.append(int(w.get("row_id", -1)))
                groups.append(gi)
                cats.append(line["category"])
        if not words:
            continue
        order = reading_order(rows, quads)
        words, quads = [words[i] for i in order], [quads[i] for i in order]
        groups, cats = [groups[i] for i in order], [cats[i] for i in order]
        size = gt["meta"]["image_size"]
        img = None
        if image != "none":
            ignore = _quads(gt.get("repeating_symbol")) + _quads(gt.get("dontcare"))
            img = mask_to_annotations(_decode(row["image"]), every, ignore)
            if image == "small":
                img = resized(img)
        yield Receipt(str(ri), words, quads, (int(size["width"]), int(size["height"])), bio(groups, cats), groups, img)


def cord_receipts(split: str = "test", root: Path | None = None, image: str = "small") -> list[Receipt]:
    return list(iter_cord(split, root, image))


def label_list(receipts: list[Receipt]) -> list[str]:
    """O, then B and I for every category seen, sorted: the same numbering for every model."""
    cats = sorted({t[2:] for r in receipts for t in r.tags if t != "O"})
    return ["O"] + [f"{p}-{c}" for c in cats for p in ("B", "I")]


def decode(words: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Fields from tags. B starts a field; I continues the latest field of its category, even after
    words of other fields in between (a field can span two rows); an I with no field of its
    category open starts one."""
    fields: list[tuple[str, list[str]]] = []
    latest: dict[str, int] = {}
    for w, t in zip(words, tags, strict=True):
        if t == "O" or len(t) < 3:
            continue
        kind, cat = t[0], t[2:]
        if kind == "I" and cat in latest:
            fields[latest[cat]][1].append(w)
        else:
            latest[cat] = len(fields)
            fields.append((cat, [w]))
    return [(c, field_text(ws)) for c, ws in fields]


def counts(true: list[tuple[str, str]], pred: list[tuple[str, str]]) -> tuple[int, int, int]:
    """True positives, false positives and false negatives of one receipt's fields. Texts are
    compared without spaces: OCR and CORD cut a line into words differently ("Sub Total" against
    "SubTotal"), and a field is its characters, not its word breaks."""
    t = Counter((c, "".join(x.split())) for c, x in true)
    p = Counter((c, "".join(x.split())) for c, x in pred)
    tp = sum((t & p).values())
    return tp, sum(p.values()) - tp, sum(t.values()) - tp


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def resized(img: np.ndarray, side: int = 224) -> np.ndarray:
    """The receipt as a side x side RGB image, for LayoutLMv3's image patches."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    return cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)


# OCR words
def ocr_receipt(rid: str, quads: list[np.ndarray], texts: list[str], size: tuple[int, int],
                image: np.ndarray | None) -> Receipt:
    """A receipt as an OCR read it: its words in the order given, empty readings left out, no tags."""
    keep = [i for i, t in enumerate(texts) if normalize(t).strip()]
    return Receipt(rid, [normalize(texts[i]).strip() for i in keep], [np.float32(quads[i]) for i in keep], size,
                   image=image)


def transfer(ocr: Receipt, truth: Receipt, share: float = 0.5) -> Receipt:
    """OCR words tagged with the true field that covers most of each (summed over that field's
    words, so an OCR word that joins two true words still finds their field), when it covers at
    least `share` of the OCR word's area; the rest are O. Tags follow the OCR words' order, so
    decoding them shows the best any tagger can do on these OCR words."""
    from .pagescore import polygon
    tp = [polygon(q) for q in truth.quads]
    groups, cats = [], []
    for q in ocr.quads:
        p = polygon(q)
        area = max(p.area, 1e-9)
        cover: dict[int, float] = {}
        first: dict[int, int] = {}
        for k, t in enumerate(tp):
            (a0, b0, a1, b1), (c0, d0, c1, d1) = t.bounds, p.bounds
            if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                g = truth.groups[k]
                cover[g] = cover.get(g, 0.0) + p.intersection(t).area / area
                first.setdefault(g, k)
        g = max(cover, key=lambda x: cover[x]) if cover else -1
        if g >= 0 and cover[g] >= share:
            groups.append(g)
            cats.append(truth.tags[first[g]][2:])
        else:
            groups.append(-1)
            cats.append("")
    return Receipt(ocr.id, ocr.words, ocr.quads, ocr.size, bio(groups, cats), groups, ocr.image)


def ocr_receipts(split: str, truths: list[Receipt], folder: Path | None = None) -> list[Receipt]:
    """The OCR words of a split (scripts/prepare_m4.py), one receipt for each true one, with the
    true receipt's size and image."""
    path = (folder or DATA / "m4") / f"ocr_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/prepare_m4.py first")
    rows = json.loads(path.read_text(encoding="utf8"))["receipts"]
    return [ocr_receipt(t.id, [np.float32(q) for q in rows[t.id]["quads"]], rows[t.id]["texts"], t.size, t.image)
            for t in truths]


def all_labels(splits=("train", "val", "test"), root: Path | None = None) -> list[str]:
    """The label list of every model: the categories of all of CORD's splits (the list of names
    only, the same whatever is trained)."""
    return label_list([r for s in splits for r in iter_cord(s, root, image="none")])
