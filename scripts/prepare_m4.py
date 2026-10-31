"""M4: the OCR words of every CORD receipt, for reading fields from OCR text.

    python scripts/prepare_m4.py                    train, val and test, to data/m4/ocr_<split>.json

docTR's pretrained detector finds the words of each receipt (masked to its annotated part, as in
M3) and M2's CRNN reads them; the words come in docTR's default order, its lines top to bottom and
each line left to right. That is M3's best reader of receipts (5.6% page CER, decision 41 in
docs/PLAN.md). A split is read again only when the detector, the reader or their code changed.
"""
import argparse
import hashlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, REPO, doctr_lines, pagereader, pages  # noqa: E402
from doclens.crnn import pick_device  # noqa: E402
from doclens.fields import iter_cord  # noqa: E402
from doclens.pagereader import read_page  # noqa: E402
from doclens.recognize import Recognizer  # noqa: E402

OUT = DATA / "m4"


def file_key(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--checkpoint", default=str(REPO / "runs" / "m2" / "main" / "best.pt"))
    ap.add_argument("--device", default="auto", help="the CRNN's device")
    ap.add_argument("--doctr-device", default="auto")
    args = ap.parse_args()
    key = (f"doctr {metadata.version('python-doctr')} db_resnet50 code {file_key(Path(doctr_lines.__file__))}"
           f" crnn {file_key(Path(args.checkpoint))} reader {file_key(Path(pagereader.__file__), Path(pages.__file__))}")
    OUT.mkdir(parents=True, exist_ok=True)
    det = rec = None
    for split in args.splits:
        path = OUT / f"ocr_{split}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf8")).get("key") == key:
            print(f"{split}: up to date ({path})")
            continue
        if det is None:
            det = doctr_lines.DoctrLines(device=str(pick_device(args.doctr_device)))
            rec = Recognizer.load(args.checkpoint, args.device)
        t, rows = time.time(), {}
        for r in iter_cord(split, image="full"):
            quads = det.words(r.image)
            texts = read_page(rec, r.image, quads) if quads else []
            rows[r.id] = {"quads": [q.round(1).tolist() for q in quads], "texts": texts}
            if len(rows) % 100 == 0:
                print(f"   {split}: {len(rows)} receipts ({(time.time() - t) / len(rows):.2f} s each)", flush=True)
        path.write_text(json.dumps({"key": key, "receipts": rows}, ensure_ascii=False), encoding="utf8")
        words = sum(len(v["texts"]) for v in rows.values())
        print(f"{split}: {len(rows)} receipts, {words} words read in {time.time() - t:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
