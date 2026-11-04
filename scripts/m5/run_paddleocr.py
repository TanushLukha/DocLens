"""M5 runner for PaddleOCR 3.7.0, in .venv_paddle (scripts/setup_m5.sh).

    .venv_paddle/bin/python scripts/m5/run_paddleocr.py --task crops
    .venv_paddle/bin/python scripts/m5/run_paddleocr.py --task pages

PaddleOCR's own choice of models for the language it is given: for English PP OCRv6 (its medium
detector and recogniser), for Hindi PP OCRv5 (its server detector and its Devanagari recogniser,
whose 568 characters hold the Latin alphabet too, so it also reads the mixed sets). Pages: the
PaddleOCR pipeline as its quick start shows it, without the document orientation, unwarping and
text line orientation models, its lines in its own order (top to bottom, then left to right).
Crops: the same recogniser alone (TextRecognition). PaddlePaddle has no Apple GPU backend, so it
runs on the CPU. There its detector cannot read a page at the size it is given by default (up to
4,000 pixels a side): without oneDNN, PaddlePaddle's convolutions on the Mac's CPU unroll every
image in full, and the large kernels of PP OCRv5's server detector then need about 45 GB for a
300 dpi page, so the first page was killed. The detector here sees a page at most DET_SIDE pixels
a side, the largest size scripts/m5/probe.py measured (960, 1280 and 1600 all found every line of
its page, with peaks of 6.2, 8.3 and 10.0 GB; decision 48 in docs/PLAN.md); the recogniser still
reads each line from the full page. On the Mac PaddlePaddle takes CPU memory from a buddy allocator
that keeps what it took, so scripts/run_m5.sh reads each set of pages in a process of its own, and
the pages are read largest first, so later ones fit in what the first took.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

LANG = {"hi": "hi", "hi+en": "hi", "en": "en"}
DET_SIDE = 1600
# the recognisers PaddleOCR 3.7.0 picks for these languages (paddleocr/_pipelines/ocr.py)
REC = {"hi": "devanagari_PP-OCRv5_mobile_rec", "en": "PP-OCRv6_medium_rec"}


def three(gray):
    import numpy as np
    return np.repeat(gray[:, :, None], 3, axis=2)


def extra(ap):
    ap.add_argument("--det-side", type=int, default=DET_SIDE, help="the detector's largest input side, in pixels")


def main():
    a = common.args(__doc__, extra)
    import numpy as np
    from paddleocr import PaddleOCR, TextRecognition

    rows = common.pick(common.load(a.task), a.sets, a.limit)
    tag = f"paddleocr {common.versions(['paddleocr'])['paddleocr']} defaults for the language, no preprocessing"
    if a.task == "pages":
        tag += f", detector input at most {a.det_side} pixels a side"
    results = common.Results("paddleocr", a.task, tag)
    models, load_s = {}, []

    def model(lang):
        key = LANG[lang]
        if key not in models:
            t = time.perf_counter()
            if a.task == "crops":
                models[key] = TextRecognition(model_name=REC[key], device="cpu")
            else:
                models[key] = PaddleOCR(lang=key, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                        use_textline_orientation=False, device="cpu", text_det_limit_type="max",
                                        text_det_limit_side_len=a.det_side)
            load_s.append(time.perf_counter() - t)
        return models[key]

    def read_crops(part):
        out = model(part[0]["lang"]).predict(input=[three(common.array(r)) for r in part], batch_size=len(part))
        return [{"text": res["rec_text"]} for res in out]

    def read_pages(part):
        out = []
        for r in part:
            (res,) = model(r["lang"]).predict(three(common.array(r)))
            lines = [{"quad": np.asarray(q, np.float32).round(1).tolist(), "text": t}
                     for q, t in zip(res["rec_polys"], res["rec_texts"])]
            out.append({"text": "\n".join(line["text"] for line in lines), "lines": lines})
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"paddleocr {a.task} {name}: {len(rs)} images, lang {LANG[rs[0]['lang']]}", flush=True)
        model(rs[0]["lang"])                     # loaded before the clock starts
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=16, show=a.show)
        else:
            def area(r):                         # the detector's input, in pixels
                w, h = r["size"]
                return w * h * min(1.0, a.det_side / max(w, h)) ** 2
            n += common.run(results, sorted(rs, key=area, reverse=True), read_pages, batch=1, show=a.show, every=25)
    info = {"tag": tag, "read": n, "load_seconds": sum(load_s), "memory": common.memory(), "device": "cpu",
            "models": REC if a.task == "crops" else "PaddleOCR pipeline for lang hi and en",
            "det_side": a.det_side if a.task == "pages" else None,
            "versions": common.versions(["paddleocr", "paddlex", "paddlepaddle", "numpy", "opencv_contrib_python"])}
    print(f"paddleocr {a.task}: {n} read; meta in {common.record('paddleocr', a.task, info)}")


if __name__ == "__main__":
    main()
