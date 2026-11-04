"""M5: two checks on a few exported images before the full run, after the first smoke run.

    bash scripts/m5/probe.sh            runs every part in the environment it needs, then the report

The smoke run raised two questions, answered here with measurements, not guesses:

vl_server, vl_hf: PaddleOCR VL read clean Hindi crops as nonsense on the Mac. Is that the model, or
  how the Mac serves it? PaddleOCR's pipeline sends MLX VLM each block as a JPEG (PNG only to
  llama.cpp), and MLX VLM cannot be told the pipeline's sizes. vl_server asks the running MLX VLM
  server directly, the crop sent as JPEG or PNG, as it is or first enlarged as the pipeline enlarges
  a block (to at least 112,896 pixels); vl_hf reads the same crops with the model in Hugging Face
  Transformers, sized as the pipeline sizes a block, on the Apple GPU and on the CPU.
paddle_det: PaddleOCR's page pipeline for Hindi was killed for lack of memory on the first page,
  which its detector reads whole (up to 4,000 pixels a side). Here one synthetic page is read with
  the detector's input limited to 960, 1280 and 1600 pixels a side, each in a fresh process: peak
  memory, seconds and the text.
report: CER of every variant (M1's scorer), from what the others wrote to runs/m5_probe.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

OUT = common.REPO / "runs" / "m5_probe"
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
SETS = {"mozhi": 10, "synth": 10, "cord_line": 10}
MIN_PIXELS, MAX_PIXELS = 112896, 1003520          # the pipeline's sizes for a text block
SIDES = (960, 1280, 1600)


def crops() -> list[dict]:
    rows = common.load("crops")
    return [r for name, n in SETS.items() for r in [x for x in rows if x["set"] == name][:n]]


def smart_resize(h: int, w: int, factor: int = 28, lo: int = MIN_PIXELS, hi: int = MAX_PIXELS) -> tuple[int, int]:
    """The size the model's own processor gives an image (Qwen2 VL's rule, as in Transformers)."""
    if h < factor:
        w, h = round(w * factor / h), factor
    if w < factor:
        h, w = round(h * factor / w), factor
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > hi:
        beta = math.sqrt(h * w / hi)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < lo:
        beta = math.sqrt(lo / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def encoded(img, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)                     # Pillow's default JPEG quality, as PaddleOCR's pipeline sends it
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ask(server: str, url: str, prompt: str = "OCR:", tokens: int = 256) -> str:
    body = {"model": MODEL, "temperature": 0, "max_tokens": tokens,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request(server.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def vl_server(a) -> None:
    from PIL import Image
    out = {}
    for variant, fmt, big in (("jpeg", "JPEG", False), ("png", "PNG", False), ("jpeg, enlarged", "JPEG", True),
                              ("png, enlarged", "PNG", True)):
        rows, t0 = [], time.perf_counter()
        for r in crops():
            img = common.image(r, "RGB")
            if big:
                h, w = smart_resize(img.height, img.width)
                img = img.resize((w, h), Image.BICUBIC)
            rows.append({"id": r["id"], "text": ask(a.server, encoded(img, fmt))})
        out[f"MLX VLM, {variant}"] = {"rows": rows, "seconds": (time.perf_counter() - t0) / len(rows)}
        print(f"MLX VLM, {variant}: {out[f'MLX VLM, {variant}']['seconds']:.2f} s each", flush=True)
    write("vl_server", out)


def vl_hf(a) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    out = {}
    for device, dtype in (("mps", torch.bfloat16), ("cpu", torch.float32)):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        name = f"Transformers, {'Apple GPU, bfloat16' if device == 'mps' else 'CPU, float32'}"
        rows, t0 = [], time.perf_counter()
        try:
            model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dtype).to(device).eval()
            todo = crops() if device == "mps" else [r for r in crops() if r["set"] != "cord_line"]
            for r in todo:
                msgs = [{"role": "user", "content": [{"type": "image", "image": common.image(r, "RGB")},
                                                     {"type": "text", "text": "OCR:"}]}]
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                  return_tensors="pt",
                                                  images_kwargs={"size": {"shortest_edge": MIN_PIXELS,
                                                                          "longest_edge": MAX_PIXELS}}).to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                rows.append({"id": r["id"],
                             "text": proc.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)})
            del model
        except Exception as e:                    # one device failing must not hide the other's readings
            print(f"{name}: failed, {type(e).__name__}: {e}"[:400], flush=True)
            if not rows:
                continue
        out[name] = {"rows": rows, "seconds": (time.perf_counter() - t0) / len(rows)}
        print(f"{name}: {out[name]['seconds']:.2f} s each", flush=True)
    write("vl_hf", out)


def paddle_one(a) -> None:
    """One page through PaddleOCR in this fresh process; prints one JSON line."""
    import numpy as np
    from paddleocr import PaddleOCR
    row = next(r for r in common.load("pages") if r["id"] == a.page)
    gray = common.array(row)
    ocr = PaddleOCR(lang="hi", use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, device="cpu", text_det_limit_type="max",
                    text_det_limit_side_len=a.side)
    t0 = time.perf_counter()
    (res,) = ocr.predict(np.repeat(gray[:, :, None], 3, axis=2))
    print(json.dumps({"side": a.side, "seconds": time.perf_counter() - t0, "lines": len(res["rec_texts"]),
                      "peak": common.memory()["peak"], "text": "\n".join(res["rec_texts"])}, ensure_ascii=False))


def paddle_det(a) -> None:
    out = {}
    for side in SIDES:
        p = subprocess.run([sys.executable, __file__, "paddle_one", "--side", str(side), "--page", a.page],
                           capture_output=True, text=True, timeout=1800)
        lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
        if p.returncode or not lines:
            out[str(side)] = {"side": side, "failed": f"exit {p.returncode}", "log": p.stderr[-800:]}
            print(f"{side}: failed, exit {p.returncode}", flush=True)
            break                                 # a larger input needs more memory still
        out[str(side)] = json.loads(lines[-1])
        o = out[str(side)]
        print(f"{side}: {o['lines']} lines, {o['seconds']:.1f} s, peak {o['peak'] / 2**30:.1f} GB", flush=True)
    write("paddle_det", {"page": a.page, "sides": out})


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf8")


def report(a) -> None:
    sys.path.insert(0, str(common.REPO / "src"))
    from doclens.metrics import Scores
    from doclens.pagescore import page_edits
    truth = {r["id"]: r for r in crops()}
    print("\nPaddleOCR VL on the same crops: CER per set, then a few readings")
    variants = {}
    for name in ("vl_server", "vl_hf"):
        p = OUT / f"{name}.json"
        if p.exists():
            variants.update(json.loads(p.read_text(encoding="utf8")))
    print(f"{'':40} " + " ".join(f"{s:>10}" for s in SETS) + "  s each")
    for v, d in variants.items():
        cells = []
        for s in SETS:
            sc = Scores()
            for row in d["rows"]:
                if truth[row["id"]]["set"] == s:
                    sc.add(truth[row["id"]]["text"], row["text"])
            cells.append(f"{100 * sc.summary()['cer']:9.1f}%" if sc.samples else f"{'':>10}")
        print(f"{v:40} " + " ".join(cells) + f"  {d['seconds']:.2f}")
    for rid in [r["id"] for r in crops()][::5]:
        print(f"\n{rid}: true {truth[rid]['text']!r}")
        for v, d in variants.items():
            got = next((row["text"] for row in d["rows"] if row["id"] == rid), None)
            if got is not None:
                print(f"   {v:38} {got[:60]!r}")
    p = OUT / "paddle_det.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf8"))
        page = next(r for r in common.load("pages") if r["id"] == d["page"])
        print(f"\nPaddleOCR on {d['page']} ({page['size'][0]} x {page['size'][1]} pixels), the detector's input limited")
        for o in d["sides"].values():
            if "failed" in o:
                print(f"   {o['side']:>5} pixels a side: failed ({o['failed']})")
                continue
            edits, chars = page_edits(page["text"], o["text"])
            print(f"   {o['side']:>5} pixels a side: {o['lines']} lines (true {len(page['lines'])}), page CER "
                  f"{100 * edits / chars:.1f}%, {o['seconds']:.1f} s, peak memory {o['peak'] / 2**30:.1f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=("vl_server", "vl_hf", "paddle_det", "paddle_one", "report"))
    ap.add_argument("--server", default="http://127.0.0.1:8111/")
    ap.add_argument("--page", default="synth/deva_1col_00_clean")
    ap.add_argument("--side", type=int, default=960)
    a = ap.parse_args()
    {"vl_server": vl_server, "vl_hf": vl_hf, "paddle_det": paddle_det, "paddle_one": paddle_one,
     "report": report}[a.part](a)


if __name__ == "__main__":
    main()
