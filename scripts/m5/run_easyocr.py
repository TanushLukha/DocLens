"""M5 runner for EasyOCR 1.7.2, in .venv_easyocr (scripts/setup_m5.sh).

    .venv_easyocr/bin/python scripts/m5/run_easyocr.py --task crops
    .venv_easyocr/bin/python scripts/m5/run_easyocr.py --task pages

EasyOCR as its README shows it: a Reader for the languages of the set (Hindi, English, or both;
its Hindi model is its Devanagari one), on the Apple GPU when PyTorch has one. Crops: recognize()
on the whole image, the recognizer alone. Pages: readtext(), its CRAFT detector and its recognizer,
each piece of text with its box, in the order EasyOCR gives them.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

LANGS = {"hi": ["hi"], "en": ["en"], "hi+en": ["hi", "en"]}


def main():
    a = common.args(__doc__)
    import easyocr
    import numpy as np

    rows = common.pick(common.load(a.task), a.sets, a.limit)
    tag = f"easyocr {common.versions(['easyocr'])['easyocr']} default settings"
    results = common.Results("easyocr", a.task, tag)
    readers, load_s = {}, []

    def reader(lang):
        if lang not in readers:
            t = time.perf_counter()
            readers[lang] = easyocr.Reader(LANGS[lang], gpu=True, verbose=False)
            load_s.append(time.perf_counter() - t)
        return readers[lang]

    def read_crops(part):
        r = reader(part[0]["lang"])
        return [{"text": " ".join(r.recognize(common.array(row), detail=0))} for row in part]

    def read_pages(part):
        out = []
        for row in part:
            found = reader(row["lang"]).readtext(common.array(row))
            lines = [{"quad": np.asarray(box, np.float32).round(1).tolist(), "text": text} for box, text, _ in found]
            out.append({"text": "\n".join(line["text"] for line in lines), "lines": lines})
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"easyocr {a.task} {name}: {len(rs)} images, {LANGS[rs[0]['lang']]}", flush=True)
        reader(rs[0]["lang"])                    # loaded before the clock starts
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=16, show=a.show)
        else:
            n += common.run(results, rs, read_pages, batch=1, show=a.show, every=25)
    device = next(iter(readers.values())).device if readers else None
    info = {"tag": tag, "read": n, "load_seconds": sum(load_s), "memory": common.memory(), "device": str(device),
            "versions": common.versions(["easyocr", "torch", "torchvision", "opencv_python_headless", "numpy"])}
    print(f"easyocr {a.task}: {n} read; meta in {common.record('easyocr', a.task, info)}")


if __name__ == "__main__":
    main()
