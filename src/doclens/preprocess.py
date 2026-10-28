"""Image clean up before OCR, with OpenCV and a little NumPy.

Each step is a function on a uint8 grayscale image. PIPELINES names the combinations the
M1 study measures, so a results table can say exactly what was done to the pixels.
Sauvola binarisation and the deskew search are written out rather than taken from a
library, because they are the two steps whose behaviour is worth understanding.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .degrade import rotate

Image = np.ndarray


def gray(img: Image) -> Image:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def scale_to_height(img: Image, height: int = 64) -> Image:
    """Resize so the image is `height` pixels tall, keeping the aspect ratio.

    For a single line this puts the letters near the size Tesseract reads best
    (an x height of roughly 20 to 30 pixels).
    """
    h, w = img.shape[:2]
    if h == height:
        return img
    f = height / h
    interp = cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA
    return cv2.resize(img, (max(1, round(w * f)), height), interpolation=interp)


def denoise(img: Image) -> Image:
    """A 3 x 3 median: removes isolated noisy pixels, keeps stroke edges."""
    return cv2.medianBlur(img, 3)


def clahe(img: Image) -> Image:
    """Contrast limited adaptive histogram equalisation: evens out uneven light."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def otsu(img: Image) -> Image:
    """One global threshold, chosen to separate the two peaks of the histogram."""
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def sauvola(img: Image, window: int = 31, k: float = 0.2, r: float = 128.0) -> Image:
    """Sauvola's local threshold: T = m * (1 + k * (s / r minus 1)).

    m and s are the mean and standard deviation in a window around each pixel. Where the
    window is flat (plain paper) s is small and T drops below m, so faint smudges stay
    white; across a stroke s is large and T sits near m. One global threshold cannot do
    both when the light is uneven. Box filters compute every local mean in O(1) per pixel,
    the same trick as an integral image.
    """
    f = img.astype(np.float64)
    m = cv2.boxFilter(f, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(f * f, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    t = m * (1.0 + k * (s / r - 1.0))
    return np.where(f > t, 255, 0).astype(np.uint8)


def _ink(img: Image) -> np.ndarray:
    """Coordinates (x, y) of the dark pixels, found with Otsu on a working copy."""
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def skew_angle(img: Image, max_angle: float = 8.0) -> float:
    """The rotation, in degrees, that makes the text rows sharpest.

    For each candidate angle the ink pixels are rotated and counted per row; when the rows
    of text are level, the counts form tall narrow peaks, so the sum of squared counts is
    largest. A coarse search in steps of 0.5 degrees is refined in steps of 0.05.
    Rotating coordinates instead of images makes each candidate cheap.
    """
    pts = _ink(img)
    if len(pts) < 20:
        return 0.0
    pts -= pts.mean(axis=0)

    def sharpness(deg: float) -> float:
        a = np.deg2rad(deg)
        y = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
        counts = np.bincount(np.round(y - y.min()).astype(np.int64))
        return float(np.sum(counts.astype(np.float64) ** 2))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = max(coarse, key=sharpness)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05)
    return float(max(fine, key=sharpness))


def deskew(img: Image, max_angle: float = 8.0) -> Image:
    a = skew_angle(img, max_angle)
    return img if abs(a) < 0.1 else rotate(img, a)


Step = Callable[[Image], Image]
PIPELINES: dict[str, list[Step]] = {
    "raw": [],
    "scale": [gray, scale_to_height],
    "scale+denoise": [gray, scale_to_height, denoise],
    "scale+clahe": [gray, scale_to_height, clahe],
    "scale+otsu": [gray, scale_to_height, otsu],
    "scale+sauvola": [gray, scale_to_height, sauvola],
    "scale+deskew": [gray, scale_to_height, deskew],
    "full": [gray, scale_to_height, deskew, denoise, sauvola],
}


def apply(name: str, img: Image) -> Image:
    for step in PIPELINES[name]:
        img = step(img)
    return img
