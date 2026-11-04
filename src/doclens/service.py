"""The DocLens service (M6): pages and receipts read over HTTP.

    uvicorn doclens.service:app --host 127.0.0.1 --port 8000

    POST /page       an image of a printed page (Hindi, English or both): its lines in reading order
    POST /receipt    an image of a receipt: its words, and its fields with the checks of their sums
    GET  /health     which models are loaded
    GET  /           a page to try it in the browser

Models come from the folder named by DOCLENS_MODELS (default: models/ in the working folder; see
doclens.pipeline). An image is sent as a file in a multipart form, field name "image", in any
format OpenCV reads (PNG, JPEG, TIFF, BMP, WebP); it is read in grey, as everything was measured.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import __version__
from .pipeline import DocLens

MAX_BYTES = 25 * 2**20                   # a 300 dpi A4 page as PNG is about 2 MB
MAX_PIXELS = 60_000_000                  # an A3 page at 600 dpi is 70 million; anything larger is refused
DEMO = Path(__file__).resolve().parent / "demo.html"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    threads = int(os.environ["DOCLENS_THREADS"]) if os.environ.get("DOCLENS_THREADS") else None
    STATE["doclens"] = DocLens(os.environ.get("DOCLENS_MODELS", "models"), threads)
    yield
    STATE.clear()


app = FastAPI(title="DocLens", version=__version__, lifespan=lifespan,
              description="Reads printed Hindi and English pages, and the fields of receipts.")


def _image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "the image is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"the image is larger than {MAX_BYTES // 2**20} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "this is not an image OpenCV can read")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "the image has more than 60 million pixels")
    if min(img.shape[:2]) < 16:
        raise HTTPException(400, "the image is smaller than 16 pixels on a side")
    return img


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "models": STATE["doclens"].info()}


@app.post("/page")
def page(image: Annotated[UploadFile, File()]) -> dict:
    return STATE["doclens"].page(_image(image.file.read()))


@app.post("/receipt")
def receipt(image: Annotated[UploadFile, File()], fields: bool = True) -> dict:
    return STATE["doclens"].receipt(_image(image.file.read()), fields)


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(DEMO, media_type="text/html")
