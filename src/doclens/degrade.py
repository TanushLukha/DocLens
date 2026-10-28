"""Ways a printed line gets damaged between the printer and the OCR engine.

Each condition takes a uint8 grayscale image and a random generator and returns a new image.
Blur, noise and low DPI use the settings of a published Devanagari OCR stress test (Gaussian blur with
sigma 1.0 to 1.8, noise on 6% of the pixels, a 0.45 downscale then a bilinear upscale), so
our numbers sit next to theirs. Skew, shadow, JPEG and a phone photo combination are
added because scans and phone pictures of documents fail in those ways too.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

Image = np.ndarray


def blur(img: Image, rng: np.random.Generator, sigma: tuple[float, float] = (1.0, 1.8)) -> Image:
    return cv2.GaussianBlur(img, (0, 0), float(rng.uniform(*sigma)))


def noise(img: Image, rng: np.random.Generator, frac: float = 0.06, std: float = 64.0) -> Image:
    out = img.astype(np.float32)
    mask = rng.random(img.shape) < frac
    out[mask] += rng.normal(0.0, std, int(mask.sum()))
    return np.clip(out, 0, 255).astype(np.uint8)


def low_dpi(img: Image, rng: np.random.Generator, scale: float = 0.45) -> Image:
    h, w = img.shape
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg(img: Image, rng: np.random.Generator, quality: tuple[int, int] = (15, 30)) -> Image:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*quality))])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def rotate(img: Image, angle: float, fill: int | None = None) -> Image:
    """Rotate about the centre, growing the canvas so nothing is cut off."""
    h, w = img.shape
    fill = int(np.median(img)) if fill is None else fill
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=fill)


def skew(img: Image, rng: np.random.Generator, degrees: tuple[float, float] = (2.0, 5.0)) -> Image:
    return rotate(img, float(rng.uniform(*degrees)) * (1 if rng.random() < 0.5 else -1))


def shadow(img: Image, rng: np.random.Generator, depth: tuple[float, float] = (0.35, 0.6)) -> Image:
    """A soft light falloff across the image, as under a desk lamp or a hand."""
    h, w = img.shape
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    gain = 1.0 - float(rng.uniform(*depth)) * ramp
    return np.clip(img.astype(np.float32) * gain[None, :], 0, 255).astype(np.uint8)


def perspective(img: Image, rng: np.random.Generator, strength: float = 0.06) -> Image:
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-strength, strength, (4, 2)) * np.float32([w * 0.25, h])
    dst = (src + jitter).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(np.median(img)))


def photo(img: Image, rng: np.random.Generator) -> Image:
    """A phone picture of a page: slight perspective, uneven light, soft focus, JPEG."""
    img = perspective(img, rng, 0.04)
    img = shadow(img, rng, (0.25, 0.45))
    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.6, 1.0)))
    return jpeg(img, rng, (35, 55))


CONDITIONS: dict[str, Callable[[Image, np.random.Generator], Image]] = {
    "clean": lambda img, rng: img,
    "blur": blur,
    "noise": noise,
    "low_dpi": low_dpi,
    "jpeg": jpeg,
    "skew": skew,
    "shadow": shadow,
    "photo": photo,
}
