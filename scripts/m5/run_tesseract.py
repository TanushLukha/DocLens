"""M5 runner for Tesseract, in the main environment.

    python scripts/m5/run_tesseract.py --task crops
    python scripts/m5/run_tesseract.py --task pages

Tesseract as M1 and M3 ran it: the best LSTM models, each set's language (hin, eng or hin+eng),
page mode 7 for a crop (one line) and 3 for a page (Tesseract finds the lines itself), one process
per core. On pages every word comes out with its own box, in Tesseract's order. Memory is the
largest peak of any one Tesseract process; as many run at once as there are cores.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import common  # noqa: E402

from doclens import DATA, tesseract  # noqa: E402

LANGS = {"hi": "hin", "en": "eng", "hi+en": "hin+eng"}


def extra(ap):
    ap.add_argument("--workers", type=int, default=os.cpu_count())


def main():
    a = common.args(__doc__, extra)
    rows = common.pick(common.load(a.task), a.sets, a.limit)
    psm = 7 if a.task == "crops" else 3
    tag = f"{tesseract.version()} best psm {psm}"
    results = common.Results("tesseract", a.task, tag)

    def cfg(row):
        return tesseract.TessConfig(LANGS[row["lang"]], psm, tessdata=str(DATA / "tessdata_best"))

    def read_crops(part):
        return [{"text": t} for t in tesseract.recognize([common.array(r) for r in part], cfg(part[0]), a.workers)]

    def read_pages(part):
        found = tesseract.read_pages([common.array(r) for r in part], cfg(part[0]), a.workers, words=True)
        return [{"text": " ".join(f.texts),
                 "lines": [{"quad": q.round(1).tolist(), "text": t} for q, t in zip(f.quads, f.texts)]} for f in found]

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"tesseract {a.task} {name}: {len(rs)} images, {LANGS[rs[0]['lang']]}", flush=True)
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=400, show=a.show)
        else:
            n += common.run(results, rs, read_pages, batch=2 * a.workers, show=a.show, every=2 * a.workers)
    info = {"tag": tag, "read": n, "load_seconds": 0.0, "workers": a.workers,
            "memory": {"now": None, "peak": common.children_peak()}, "memory_note": "largest single process",
            "versions": {"tesseract": tesseract.version()}, "device": "cpu"}
    print(f"tesseract {a.task}: {n} read; meta in {common.record('tesseract', a.task, info)}")


if __name__ == "__main__":
    main()
