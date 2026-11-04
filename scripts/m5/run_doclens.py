"""M5 runner for DocLens, in the main environment.

    python scripts/m5/run_doclens.py --task crops
    python scripts/m5/run_doclens.py --task pages

Crops: M2's CRNN reads each image, greedy decoding, 64 at a time on the GPU (the setting the
service ships; M2's beam search with TinyGPT moved CER by at most 0.13 points). Pages: M3's best
finder for each kind of page, the classical line finder on synthetic pages and docTR's detector on
receipts, its words in docTR's top to bottom order as M4 read them; the CRNN reads every line or
word. DocLens is never told the language.
"""
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import common  # noqa: E402

from doclens import REPO  # noqa: E402


def digest(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def extra(ap):
    ap.add_argument("--checkpoint", default=str(REPO / "runs" / "m2" / "main" / "best.pt"))
    ap.add_argument("--device", default="auto", help="the CRNN's device")
    ap.add_argument("--doctr-device", default="auto")


def main():
    a = common.args(__doc__, extra)
    import torch

    from doclens import crnn, doctr_lines, layout, pagereader, pages, recognize
    from doclens.crnn import pick_device
    from doclens.layout import LineFinder
    from doclens.pagereader import read_page
    from doclens.recognize import Recognizer

    rows = common.pick(common.load(a.task), a.sets, a.limit)
    code = [Path(m.__file__) for m in (crnn, recognize, pagereader, pages, layout, doctr_lines)]
    tag = f"crnn {digest(Path(a.checkpoint))} code {digest(*code)} greedy batch 64"
    results = common.Results("doclens", a.task, tag)
    t0 = time.perf_counter()
    rec = Recognizer.load(a.checkpoint, a.device)
    finder = LineFinder()
    det = []                                    # docTR's detector, loaded for the receipts
    load_s = [time.perf_counter() - t0]

    def read_crops(part):
        return [{"text": t} for t in rec.read([common.array(r) for r in part], batch_size=64)]

    def read_pages(part):
        out = []
        for r in part:
            img = common.array(r)
            if r["set"] == "cord":
                quads, sep = det[0].words(img), " "
            else:
                quads, sep = finder(img), "\n"
            texts = read_page(rec, img, quads) if quads else []
            out.append({"text": sep.join(texts),
                        "lines": [{"quad": q.round(1).tolist(), "text": t} for q, t in zip(quads, texts)]})
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"doclens {a.task} {name}: {len(rs)} images", flush=True)
        if a.task == "pages" and name == "cord" and not det:     # loaded before the clock starts
            t = time.perf_counter()
            det.append(doctr_lines.DoctrLines(device=str(pick_device(a.doctr_device))))
            load_s.append(time.perf_counter() - t)
        if a.task == "crops":
            n += common.run(results, rs, read_crops, batch=64, show=a.show)
        else:
            n += common.run(results, rs, read_pages, batch=1, show=a.show, every=25)
    info = {"tag": tag, "read": n, "load_seconds": sum(load_s), "memory": common.memory(), "device": str(rec.device),
            "versions": common.versions(["torch", "python_doctr", "numpy", "opencv_python"]),
            "torch_mps": bool(torch.backends.mps.is_available())}
    print(f"doclens {a.task}: {n} read; meta in {common.record('doclens', a.task, info)}")


if __name__ == "__main__":
    main()
