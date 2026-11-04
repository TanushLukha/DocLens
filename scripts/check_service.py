"""M6: an end to end check of a running DocLens service, over HTTP only.

    python scripts/check_service.py --url http://127.0.0.1:8000

Sends a page and a receipt (from data/m5 when it is there, else images drawn here), an empty
upload and a file that is not an image, and checks every answer's status and shape: lines and words
with four corner boxes inside the image and their text, fields and checks for a receipt when the
service has its extractor, the demo page at /. Only the standard library, numpy and OpenCV are
used, so it runs against the Docker image from any machine. Exits with 1 on the first failure.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "m5"


def request(url: str, data: bytes | None = None, fields: dict | None = None) -> tuple[int, bytes]:
    headers = {}
    if fields is not None:                       # multipart form, one file field
        boundary = uuid.uuid4().hex
        name, (filename, payload, ctype) = next(iter(fields.items()))
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def sample(kind: str) -> np.ndarray:
    """A synthetic page or a CORD receipt from M5's export, or else an image drawn here."""
    rows = DATA / "pages.jsonl"
    if rows.exists():
        want = "synth" if kind == "page" else "cord"
        row = next(json.loads(x) for x in rows.read_text(encoding="utf8").splitlines() if f'"set": "{want}"' in x)
        img = cv2.imread(str(DATA / row["file"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    img = np.full((500, 700), 255, np.uint8)
    for k, text in enumerate(["DocLens reads this line", "and this one", "TOTAL 9,100"]):
        cv2.putText(img, text, (40, 90 + 110 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    return img


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        sys.exit(1)


def boxed(items: list, w: int, h: int) -> bool:
    return all(isinstance(it.get("text"), str) and len(it["quad"]) == 4
               and all(-2 <= x <= w + 2 and -2 <= y <= h + 2 for x, y in it["quad"]) for it in items)


def run_checks(base: str) -> None:
    """Every check against the service at `base`; exits with 1 on the first that fails."""
    base = base.rstrip("/")
    code, body = request(base + "/health")
    health = json.loads(body) if code == 200 else {}
    check(code == 200 and health.get("status") == "ok", f"/health answers: {health.get('models')}")
    code, body = request(base + "/")
    check(code == 200 and b"DocLens" in body, "the demo page is served at /")

    for kind in ("page", "receipt"):
        img = sample(kind)
        h, w = img.shape[:2]
        code, body = request(f"{base}/{kind}", fields={"image": (f"{kind}.png", cv2.imencode(".png", img)[1].tobytes(),
                                                                 "image/png")})
        check(code == 200, f"/{kind} reads a {w} x {h} image")
        r = json.loads(body)
        items = r["lines"] if kind == "page" else r["words"]
        check(r["kind"] == kind and boxed(items, w, h), f"{len(items)} {'lines' if kind == 'page' else 'words'}, each with its"
              " text and four corners inside the image")
        sep = "\n" if kind == "page" else " "
        check(r["text"] == sep.join(it["text"] for it in items if it["text"] or kind == "page"),
              "the text is its pieces in order")
        if kind == "receipt" and health["models"].get("field_extractor"):
            check(isinstance(r.get("fields"), list) and isinstance(r.get("checks"), list),
                  f"{len(r['fields'])} fields and {len(r['checks'])} checks of the sums")
        print(f"     first: {(items[0]['text'] if items else '')[:60]!r}; seconds {r['seconds']}")

    for payload, what in ((b"", "an empty upload"), (b"not an image", "a file that is not an image")):
        code, _ = request(base + "/page", fields={"image": ("x.png", payload, "image/png")})
        check(code == 400, f"{what} is refused with 400")
    print("the service works end to end")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    run_checks(ap.parse_args().url)


if __name__ == "__main__":
    main()
