"""M5 runner for Surya 0.22.1, in .venv_surya (scripts/setup_m5.sh).

    .venv_surya/bin/python scripts/m5/run_surya.py --task crops
    .venv_surya/bin/python scripts/m5/run_surya.py --task pages

Surya 2 is a vision language model; on a Mac, Surya runs it in llama.cpp's server on the Apple GPU
(Metal), which Surya starts and stops itself, with its default eight requests at a time. Pages: its
default full page OCR, one request per page that returns the page's blocks in reading order, each
with its box and its text as HTML; a page whose answer loops or cannot be parsed is read again
block by block after Surya's layout model, and what Surya logs about it is kept with the pages of
that batch. Crops: its block OCR, the whole crop as one text block. Surya is never told the
language. Memory adds the peak of llama.cpp's server to this process's.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

TOKENS = 400        # a crop's answer budget: Surya gives a block count + 100 tokens


class Notes(logging.Handler):
    """What Surya logs, kept for the batch being read (its pages that fell back, and why)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(f"{record.levelname}: {record.getMessage()}")


def main():
    a = common.args(__doc__)
    from surya.inference import SuryaInferenceManager
    from surya.layout.schema import LayoutBox, LayoutResult
    from surya.recognition import RecognitionPredictor
    from surya.settings import settings

    notes = Notes()
    log = logging.getLogger("surya")
    log.addHandler(notes)
    log.setLevel(logging.INFO)
    rows = common.pick(common.load(a.task), a.sets, a.limit)
    tag = f"surya {common.versions(['surya_ocr'])['surya_ocr']} {settings.SURYA_GGUF_MODEL_FILE} full page"
    results = common.Results("surya", a.task, tag)
    t0 = time.perf_counter()
    manager = SuryaInferenceManager(lazy=False)             # starts llama.cpp's server now
    rec = RecognitionPredictor(manager)
    load_s = time.perf_counter() - t0

    def whole(img):
        w, h = img.size
        box = LayoutBox(polygon=[[0, 0], [w, 0], [w, h], [0, h]], label="Text", raw_label="Text", position=0,
                        count=TOKENS - 100)
        return LayoutResult(bboxes=[box], image_bbox=[0, 0, w, h])

    def read_crops(part):
        imgs = [common.image(r, "RGB") for r in part]
        pages = rec(imgs, [whole(im) for im in imgs], full_page=False)
        return [{"text": " ".join(common.html_text(p.blocks[0].html).split()), "error": p.blocks[0].error}
                for p in pages]

    def read_pages(part):
        out = []
        notes.lines.clear()
        pages = rec([common.image(r, "RGB") for r in part], full_page=True)
        said = [x for x in notes.lines if "fallback" in x or "regenerat" in x]
        for x in said:
            print(f"   Surya: {x}"[:200], flush=True)
        for page in pages:
            blocks = sorted((b for b in page.blocks if not b.skipped), key=lambda b: b.reading_order)
            lines = [{"quad": [[round(x, 1), round(y, 1)] for x, y in b.polygon], "text": common.html_text(b.html),
                      "label": b.label} for b in blocks]
            out.append({"text": "\n".join(line["text"] for line in lines if line["text"]), "blocks": lines}
                       | ({"surya_log": said} if said else {}))
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"surya {a.task} {name}: {len(rs)} images", flush=True)
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=32, show=a.show)
        else:
            n += common.run(results, rs, read_pages, batch=8, show=a.show, every=8)
    servers = {p: common.memory(p) for p in common.pids("llama-server")}
    info = {"tag": tag, "read": n, "load_seconds": load_s, "memory": common.memory(), "servers": servers,
            "backend": manager.method, "parallel": manager.capacity(), "device": "Apple GPU through llama.cpp",
            "versions": common.versions(["surya_ocr", "torch", "transformers", "pillow", "openai"])}
    print(f"surya {a.task}: {n} read; meta in {common.record('surya', a.task, info)}")


if __name__ == "__main__":
    main()
