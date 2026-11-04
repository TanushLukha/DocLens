"""DocLens end to end (M6): an image of a page or a receipt in; its text, and a receipt's fields, out.

Pages: the classical line finder, then the CRNN reads every line in the finder's order (M3's best
reader of printed pages). Receipts: docTR's detector finds the words in its top to bottom order,
the CRNN reads them, LiLT tags them with CORD's fields (M4), and the receipt's sums are checked
(doclens.checks): a failing rule is reported, nothing is changed, because a check cannot say
reliably which digit is wrong (docs/results/m6_checks.md). The caller says which kind of document
it sends: the two need different finders and orders (M3), and a wrong guess costs far more than
asking. The CRNN runs through ONNX Runtime on the CPU; docTR and LiLT through PyTorch on the CPU.

A models folder (scripts/export_models.py makes it) holds crnn.onnx with crnn.json, and lilt/,
the field extractor, which may be left out: receipts then come back without fields.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from .checks import rules
from .fields import decode, ocr_receipt
from .layout import LineFinder
from .onnx_crnn import OnnxRecognizer
from .pagereader import read_page


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _quad(q) -> list[list[float]]:
    return np.asarray(q, np.float32).round(1).tolist()


class DocLens:
    def __init__(self, models: str | Path, threads: int | None = None):
        self.models = Path(models)
        self.rec = OnnxRecognizer(self.models / "crnn.onnx", threads)
        self.finder = LineFinder()
        self._detector = None                    # docTR, loaded with the first receipt
        self._extractor: tuple | None = None
        self._lock = threading.Lock()            # one image at a time: each step already uses every core

    def info(self) -> dict:
        lilt = self.models / "lilt"
        return {"crnn": _digest(self.models / "crnn.onnx"),
                "field_extractor": "lilt" if (lilt / "config.json").exists() else None}

    def detector(self):
        if self._detector is None:
            from .doctr_lines import DoctrLines
            self._detector = DoctrLines(device="cpu")
        return self._detector

    def extractor(self) -> tuple | None:
        """(model, tokenizer, labels) of the field extractor, or None when the models folder has none."""
        folder = self.models / "lilt"
        if self._extractor is None and (folder / "config.json").exists():
            from transformers import LiltForTokenClassification

            from .kie import tokenizer
            model = LiltForTokenClassification.from_pretrained(folder).eval()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            self._extractor = (model, tokenizer(str(folder)), labels)
        return self._extractor

    def page(self, img: np.ndarray) -> dict:
        """A printed page: its lines in reading order, each with its box and text."""
        with self._lock:
            t0 = time.perf_counter()
            quads = self.finder(img)
            t1 = time.perf_counter()
            texts = read_page(self.rec, img, quads) if quads else []
            t2 = time.perf_counter()
        lines = [{"text": t, "quad": _quad(q)} for q, t in zip(quads, texts)]
        return {"kind": "page", "text": "\n".join(texts), "lines": lines,
                "seconds": {"find": t1 - t0, "read": t2 - t1}}

    def receipt(self, img: np.ndarray, fields: bool = True) -> dict:
        """A receipt: its words in reading order; its fields and the checks of their sums when the
        field extractor is installed."""
        from . import kie
        with self._lock:
            t0 = time.perf_counter()
            quads = self.detector().words(img)
            t1 = time.perf_counter()
            texts = read_page(self.rec, img, quads) if quads else []
            t2 = time.perf_counter()
            kept = [(t, q) for q, t in zip(quads, texts) if t.strip()]
            out: dict = {"kind": "receipt", "text": " ".join(t for t, _ in kept),
                         "words": [{"text": t, "quad": _quad(q)} for t, q in kept]}
            ext = self.extractor() if fields else None
            if ext is not None:
                model, tok, labels = ext
                h, w = img.shape[:2]
                r = ocr_receipt("receipt", quads, texts, (w, h), None)
                found: list[tuple[str, str]] = []
                if r.words:
                    ids = {t: i for i, t in enumerate(labels)}
                    tags = kie.predict(model, "lilt", [r], kie.encode(tok, r, ids), labels, "cpu")[r.id]
                    found = decode(r.words, tags)
                out["fields"] = [{"category": c, "text": t} for c, t in found]
                out["checks"] = [{"rule": x.name, "holds": x.holds} for x in rules(found)]
            t3 = time.perf_counter()
        out["seconds"] = {"find": t1 - t0, "read": t2 - t1, "fields": t3 - t2}
        return out
