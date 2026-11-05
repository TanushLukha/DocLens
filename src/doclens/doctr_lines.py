"""docTR's pretrained text detector, with docTR's own grouping into lines and reading order (M3).

docTR (Mindee, Apache 2.0) finds words with a DBNet whose backbone is a ResNet 50, trained by its
makers on their own documents, then groups the words into lines and, when asked, puts the lines in
reading order. DocLens uses the detector and that grouping exactly as docTR ships them and reads
the lines with its own CRNN: docTR's recognizers know Latin script only.

Two orders come out of one detection: docTR's reading order (keep_reading_order, which handles
columns) and docTR's default, lines sorted top to bottom and left to right. For receipts (M4) the
words themselves come out too, in the default order.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .pages import order_quad


class DoctrLines:
    def __init__(self, arch: str = "db_resnet50", device: str = "cpu", pretrained: bool = True):
        try:
            from doctr.models import detection_predictor
            from doctr.models.builder import DocumentBuilder
        except ImportError as e:
            raise ImportError('docTR is not installed: pip install -e ".[pages]"') from e
        self.det = detection_predictor(arch, pretrained=pretrained, assume_straight_pages=False,
                                       preserve_aspect_ratio=True, symmetric_pad=True, batch_size=1)
        self.det.to(device)
        self.builders = {
            "order": DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=True),
            "rows": DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=False),
        }
        self.arch = arch

    def _documents(self, page: np.ndarray, names) -> tuple[dict, tuple[int, int]]:
        """docTR's documents for one page, one per builder named; empty when nothing is found."""
        from doctr.utils.geometry import detach_scores
        rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB) if page.ndim == 2 else page
        h, w = rgb.shape[:2]
        found = list(self.det([rgb])[0].values())[0]
        if len(found) == 0:
            return {}, (h, w)
        boxes, scores = detach_scores([found])
        n = len(boxes[0])
        orientations: list[Any] = [[{"value": 0, "confidence": None}] * n]     # one list per page
        return {name: self.builders[name]([rgb], boxes, scores, [[("", 1.0)] * n], [(h, w)], orientations)
                for name in names}, (h, w)

    @staticmethod
    def _quad(geometry, w: int, h: int) -> np.ndarray:
        g = np.asarray(geometry, np.float32)
        if g.shape == (2, 2):                              # a straight box: (x0, y0), (x1, y1)
            g = np.array([[g[0, 0], g[0, 1]], [g[1, 0], g[0, 1]], [g[1, 0], g[1, 1]], [g[0, 0], g[1, 1]]],
                         np.float32)
        return order_quad(g * np.array([w, h], np.float32))

    def __call__(self, page: np.ndarray) -> dict[str, list[np.ndarray]]:
        """Line quadrilaterals in page pixels, in docTR's reading order ('order') and in its default
        order ('rows')."""
        docs, (h, w) = self._documents(page, self.builders)
        return {name: [self._quad(line.geometry, w, h) for block in docs[name].pages[0].blocks for line in block.lines]
                       if docs else [] for name in self.builders}

    def words(self, page: np.ndarray) -> list[np.ndarray]:
        """Word quadrilaterals in page pixels, in docTR's default order: its lines top to bottom,
        the words of a line left to right (M4 reads receipts this way)."""
        docs, (h, w) = self._documents(page, ["rows"])
        if not docs:
            return []
        return [self._quad(word.geometry, w, h) for block in docs["rows"].pages[0].blocks
                for line in block.lines for word in line.words]


def rows_order(quads: list[np.ndarray], size: tuple[int, int]) -> list[int]:
    """Indices of word boxes in the order docTR gives its own detections by default: grouped into
    lines, the lines top to bottom, the words of a line left to right. M5 puts every system's words
    in this order, the one DocLens reads receipts in (M4). size is the page's (width, height)."""
    from doctr.models.builder import DocumentBuilder

    if not quads:
        return []
    w, h = size
    n = len(quads)
    boxes = np.stack([np.array(q, np.float32).reshape(4, 2) for q in quads]) / np.array([w, h], np.float32)
    orientations: list[Any] = [[{"value": 0, "confidence": None}] * n]         # one list per page
    doc = DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=False)(
        [np.zeros((1, 1, 3), np.uint8)], [boxes], [np.ones(n, np.float32)], [[(str(i), 1.0) for i in range(n)]],
        [(h, w)], orientations)
    return [int(word.value) for block in doc.pages[0].blocks for line in block.lines for word in line.words]
