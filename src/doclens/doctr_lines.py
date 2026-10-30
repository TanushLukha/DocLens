"""docTR's pretrained text detector, with docTR's own grouping into lines and reading order (M3).

docTR (Mindee, Apache 2.0) finds words with a DBNet whose backbone is a ResNet 50, trained by its
makers on their own documents, then groups the words into lines and, when asked, puts the lines in
reading order. DocLens uses the detector and that grouping exactly as docTR ships them and reads
the lines with its own CRNN: docTR's recognizers know Latin script only.

Two orders come out of one detection: docTR's reading order (keep_reading_order, which handles
columns) and docTR's default, lines sorted top to bottom and left to right.
"""
from __future__ import annotations

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

    def __call__(self, page: np.ndarray) -> dict[str, list[np.ndarray]]:
        """Line quadrilaterals in page pixels, in docTR's reading order ('order') and in its default
        order ('rows')."""
        from doctr.utils.geometry import detach_scores
        rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB) if page.ndim == 2 else page
        h, w = rgb.shape[:2]
        found = list(self.det([rgb])[0].values())[0]
        if len(found) == 0:
            return {name: [] for name in self.builders}
        boxes, scores = detach_scores([found])
        n = len(boxes[0])
        out = {}
        for name, builder in self.builders.items():
            doc = builder([rgb], boxes, scores, [[("", 1.0)] * n], [(h, w)], [[{"value": 0, "confidence": None}] * n])
            quads = []
            for block in doc.pages[0].blocks:
                for line in block.lines:
                    g = np.asarray(line.geometry, np.float32)
                    if g.shape == (2, 2):                          # a straight box: (x0, y0), (x1, y1)
                        g = np.float32([[g[0, 0], g[0, 1]], [g[1, 0], g[0, 1]], [g[1, 0], g[1, 1]], [g[0, 0], g[1, 1]]])
                    quads.append(order_quad(g * np.float32([w, h])))
            out[name] = quads
        return out
