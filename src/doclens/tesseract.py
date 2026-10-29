"""Tesseract, run the fast way: many images per process, several processes in parallel.

Starting Tesseract costs about 0.1 s, longer than reading one word. So images are written
to a temporary folder and passed in chunks through a list file; Tesseract's text output
separates the images with a form feed. Each process is limited to one thread
(OMP_THREAD_LIMIT=1) and as many processes run as there are cores: Tesseract's own
threading helps little on small images, separate processes scale almost linearly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .text import normalize


@dataclass(frozen=True)
class TessConfig:
    langs: str = "hin+eng"
    psm: int = 7                  # 7: the image is one line; 8: one word; 6: a block; 3: a page
    oem: int = 1                  # 1: the LSTM recognizer only
    tessdata: str | None = None   # a folder of .traineddata files; None: Tesseract's own
    dpi: int = 300
    extra: tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        models = Path(self.tessdata).name if self.tessdata else "default"
        return f"{self.langs} psm{self.psm} {models}"


def binary() -> str:
    path = shutil.which("tesseract")
    if path is None:
        raise RuntimeError("tesseract not found. On a Mac: brew install tesseract")
    return path


def version() -> str:
    out = subprocess.run([binary(), "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def languages(tessdata: str | None = None) -> list[str]:
    cmd = [binary(), "--list-langs"] + (["--tessdata-dir", tessdata] if tessdata else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = (out.stdout + out.stderr).splitlines()
    return [l.strip() for l in lines[1:] if l.strip() and " " not in l.strip()]


def _command(cfg: TessConfig, image_or_list: str) -> list[str]:
    cmd = [binary(), image_or_list, "stdout", "--psm", str(cfg.psm), "--oem", str(cfg.oem),
           "--dpi", str(cfg.dpi), "-l", cfg.langs]
    if cfg.tessdata:
        cmd += ["--tessdata-dir", cfg.tessdata]
    for kv in cfg.extra:
        cmd += ["-c", kv]
    return cmd


def _env() -> dict:
    return {**os.environ, "OMP_THREAD_LIMIT": "1"}


def _run_one(path: str, cfg: TessConfig) -> str:
    out = subprocess.run(_command(cfg, path), capture_output=True, env=_env())
    return out.stdout.decode("utf8", errors="replace").replace("\f", "")


def _run_chunk(paths: list[str], cfg: TessConfig, tmp: Path, idx: int) -> list[str]:
    listfile = tmp / f"list_{idx:05d}.txt"
    listfile.write_text("\n".join(paths) + "\n", encoding="utf8")
    out = subprocess.run(_command(cfg, str(listfile)), capture_output=True, env=_env())
    pieces = out.stdout.decode("utf8", errors="replace").split("\f")
    if len(pieces) == len(paths) + 1 and not pieces[-1].strip():
        pieces = pieces[:-1]
    if out.returncode != 0 or len(pieces) != len(paths):
        # an unreadable image shifts every later result: redo this chunk one image at a time
        pieces = [_run_one(p, cfg) for p in paths]
    return pieces


def recognize(images: list[np.ndarray], cfg: TessConfig, workers: int | None = None,
              chunk: int = 100) -> list[str]:
    """OCR every image; returns normalised text in the same order."""
    workers = workers or os.cpu_count() or 1
    with tempfile.TemporaryDirectory(prefix="doclens_tess_") as d:
        tmp = Path(d)
        paths = []
        for i, img in enumerate(images):
            p = tmp / f"{i:07d}.png"
            if not cv2.imwrite(str(p), img):
                raise OSError(f"could not write {p}")
            paths.append(str(p))
        chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(lambda a: _run_chunk(a[1], cfg, tmp, a[0]), enumerate(chunks)))
    return [normalize(t) for part in parts for t in part]
