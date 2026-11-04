import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from doclens.onnx_crnn import OnnxRecognizer  # noqa: E402
from doclens.pages import box_quad  # noqa: E402
from doclens.recognize import Recognizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tiny_models  # noqa: E402
from tiny_models import tiny_crnn  # noqa: E402


def page_image() -> np.ndarray:
    img = np.full((400, 600), 255, np.uint8)
    for k, text in enumerate(["first line of text", "a second line", "TOTAL 9,100"]):
        cv2.putText(img, text, (40, 80 + 90 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return img


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A models folder as scripts/export_models.py makes it, with tiny models: exported once, since
    PyTorch's exporter may fix the width of a second export in one process (doclens.onnx_crnn)."""
    return tiny_models.write(tmp_path_factory.mktemp("models"))


def test_the_onnx_graph_reads_like_pytorch(models):
    model, alphabet = tiny_crnn()
    torch_rec, onnx_rec = Recognizer(model, alphabet), OnnxRecognizer(models / "crnn.onnx")
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (40, w), dtype=np.uint8) for w in (30, 300, 90, 600, 45, 1000, 160)]
    a, b = dict(torch_rec.logprobs(images, 3)), dict(onnx_rec.logprobs(images, 3))      # batches of 3 and of 1
    assert sorted(a) == sorted(b) == list(range(7))
    assert all(a[i].shape == b[i].shape and np.allclose(a[i], b[i], atol=1e-4) for i in a)
    assert torch_rec.read(images) == onnx_rec.read(images)


@pytest.fixture
def client(models, monkeypatch):
    monkeypatch.setenv("DOCLENS_MODELS", str(models))
    from doclens import pipeline, service

    class Words:                                     # stands in for docTR's detector, which downloads weights
        def words(self, img):
            return [box_quad(30, 50, 300, 95), box_quad(30, 230, 330, 275)]

    monkeypatch.setattr(pipeline.DocLens, "detector", lambda self: Words())
    with TestClient(service.app) as c:
        yield c


def test_health_names_the_models(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and len(r["models"]["crnn"]) == 16 and r["models"]["field_extractor"] == "lilt"


def test_a_page_comes_back_as_lines_in_order(client):
    r = client.post("/page", files={"image": ("page.png", png(page_image()), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "page" and len(body["lines"]) == 3
    tops = [min(y for _, y in line["quad"]) for line in body["lines"]]
    assert tops == sorted(tops) and all(len(line["quad"]) == 4 for line in body["lines"])
    assert body["text"] == "\n".join(line["text"] for line in body["lines"])


def test_a_receipt_comes_back_with_words_fields_and_checks(client):
    r = client.post("/receipt", files={"image": ("r.jpg", cv2.imencode(".jpg", page_image())[1].tobytes(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["kind"] == "receipt"
    assert len(body["words"]) <= 2 and isinstance(body["fields"], list) and isinstance(body["checks"], list)
    plain = client.post("/receipt", params={"fields": "false"}, files={"image": ("r.png", png(page_image()), "image/png")})
    assert "fields" not in plain.json()


@pytest.mark.parametrize("data, code", [(b"", 400), (b"not an image", 400), (png(np.zeros((8, 8), np.uint8)), 400)],
                         ids=["empty", "text", "tiny"])
def test_what_is_not_a_readable_image_is_refused(client, data, code):
    assert client.post("/page", files={"image": ("x.png", data, "image/png")}).status_code == code


def test_the_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocLens" in r.text
