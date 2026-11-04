"""M5 runner for PaddleOCR VL 1.6, in .venv_paddle, with its model served by MLX VLM from .venv_mlx.

    .venv_mlx/bin/mlx_vlm.server --host 127.0.0.1 --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6 &
    .venv_paddle/bin/python scripts/m5/run_paddleocr_vl.py --task crops --server-pid $!

The whole PaddleOCR VL pipeline, as its makers ask (its guide warns that the 0.9B model alone, run
through Transformers, is not the same thing): PaddleOCR's layout model (PP DocLayoutV3, on the CPU
in PaddlePaddle) finds the blocks of a page and puts them in reading order, and the vision
language model reads each block. On Apple silicon the guide serves the model with MLX VLM on the
Apple GPU (scripts/run_m5.sh starts and stops that server). Pages: the default pipeline; receipts
also through its text spotting prompt, which gives every line with its box, for the layout aware
extractor. Crops: no layout step, the whole crop read with its OCR prompt. The model is never told
the language. A block the model writes as HTML (a table: on receipts the layout model often finds
one) becomes plain text, a line per row and a space between cells, as Surya's HTML does. Memory
adds the server's peak to this process's.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"


def extra(ap):
    ap.add_argument("--server", default="http://localhost:8111/")
    ap.add_argument("--server-pid", type=int, default=None, help="the MLX VLM server, for its memory")
    ap.add_argument("--server-version", default="mlx_vlm 0.7.11")


def three(gray):
    import numpy as np
    return np.repeat(gray[:, :, None], 3, axis=2)


def plain(label: str, content: str) -> str:
    """A block's text, with a table or any other block written as HTML turned into lines."""
    if label == "table" or re.search(r"</\w+>", content):
        return common.html_text(content)
    return content


def block(b) -> tuple[str, str, list]:
    """A parsed block's label, text and box, from the result object or its dict form."""
    if isinstance(b, dict):
        return (b.get("block_label", b.get("label", "")), b.get("block_content", b.get("content", "")) or "",
                b.get("block_bbox", b.get("bbox")))
    return getattr(b, "label", ""), getattr(b, "content", "") or "", getattr(b, "bbox", None)


def main():
    a = common.args(__doc__, extra)
    import numpy as np
    from paddleocr import PaddleOCRVL

    rows = common.pick(common.load(a.task), a.sets, a.limit)
    tag = f"paddleocr {common.versions(['paddleocr'])['paddleocr']} {MODEL} {a.server_version}, HTML as text"
    results = common.Results("paddleocr_vl", a.task, tag)
    t0 = time.perf_counter()
    pipe = PaddleOCRVL(vl_rec_backend="mlx-vlm-server", vl_rec_server_url=a.server, vl_rec_api_model_name=MODEL,
                       device="cpu")
    load_s = time.perf_counter() - t0

    def read_crops(part):
        out = pipe.predict([three(common.array(r)) for r in part], use_layout_detection=False, prompt_label="ocr")
        texts = []
        for res in out:
            blocks = [block(b) for b in res["parsing_res_list"]]
            texts.append(" ".join(" ".join(plain(lab, t).split()) for lab, t, _ in blocks).strip())
        return [{"text": t} for t in texts]

    def spotting(img):
        (res,) = pipe.predict(img, use_layout_detection=False, prompt_label="spotting")
        spot = res["spotting_res"] or {}
        lines = [{"quad": np.asarray(q, np.float32).round(1).tolist(), "text": s}
                 for q, s in zip(spot.get("rec_polys", []), spot.get("rec_texts", []))]
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines}

    def read_pages(part):
        out = []
        for r in part:
            img = three(common.array(r))
            (res,) = pipe.predict(img)
            blocks = [{"label": lab, "text": plain(lab, text), "box": [float(v) for v in box] if box is not None else None}
                      | ({"html": text} if plain(lab, text) != text else {})
                      for lab, text, box in map(block, res["parsing_res_list"])]
            row = {"text": "\n".join(b["text"] for b in blocks if b["text"]), "blocks": blocks}
            out.append(row)
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"paddleocr_vl {a.task} {name}: {len(rs)} images", flush=True)
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=16, show=a.show)
        else:
            n += common.run(results, rs, read_pages, batch=1, show=a.show, every=25)
    if a.task == "pages":                        # receipts again, through the spotting prompt
        spots = common.Results("paddleocr_vl", "spotting", tag)
        cord = [r for r in rows if r["set"] == "cord"]
        print(f"paddleocr_vl spotting cord: {len(cord)} images", flush=True)
        n += common.run(spots, cord, lambda part: [spotting(three(common.array(r))) for r in part], show=a.show,
                        every=25)
    servers = {a.server_pid: common.memory(a.server_pid)} if a.server_pid else {}
    info = {"tag": tag, "read": n, "load_seconds": load_s, "memory": common.memory(), "servers": servers,
            "device": "layout on the CPU, the model on the Apple GPU through MLX VLM",
            "versions": common.versions(["paddleocr", "paddlex", "paddlepaddle", "numpy"])}
    print(f"paddleocr_vl {a.task}: {n} read; meta in {common.record('paddleocr_vl', a.task, info)}")


if __name__ == "__main__":
    main()
