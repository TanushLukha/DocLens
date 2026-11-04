"""M5: the test images every system reads, exported once as PNG files with their true text.

    python scripts/m5_export.py           data/m5/crops.jsonl, data/m5/pages.jsonl and the images

Crops (one word or line each), a fixed random sample of every M1 set but M2's training fonts:
1,000 Mozhi test words, 1,000 synthetic lines in the six held out fonts, 1,000 CORD test words and
all 791 CORD test lines. Pages: the 150 synthetic test pages in the held out fonts and CORD's 100
test receipts. Every image is exactly what M1 to M4 read: grey, Mozhi's words with their margin,
receipts masked to the part CORD annotates. Each row also says which language a careful user would
tell a system that asks (decision 44 in docs/PLAN.md). The sample is drawn with a fixed seed, so
exporting again gives the same images; the files' digest is written to data/m5/export.json.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, datasets  # noqa: E402
from doclens.pages import cord_pages, read_set  # noqa: E402

OUT = DATA / "m5"
CROP_SETS = {          # name in M5: (M1 set, the language a careful user would give)
    "mozhi": ("mozhi", "hi"),
    "synth": ("synth_unseen", "hi+en"),
    "cord_word": ("cord_word", "en"),
    "cord_line": ("cord_line", "en"),
}


def sample(n_all: int, n: int, seed: int) -> list[int]:
    """Indices of a random sample of n of n_all (all of them when n_all <= n), in their first order."""
    if n_all <= n:
        return list(range(n_all))
    return sorted(random.Random(seed).sample(range(n_all), n))


def write(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def crops(per_set: int, seed: int) -> list[dict]:
    rows = []
    for name, (source, lang) in CROP_SETS.items():
        items = datasets.TEST_SETS[source]()
        for k, i in enumerate(sample(len(items), per_set, seed)):
            it = items[i]
            file = f"crops/{name}/{k:05d}.png"
            write(OUT / file, it.image())
            rows.append({"id": f"{name}/{k:05d}", "set": name, "file": file, "text": it.text, "lang": lang,
                         "meta": {**it.meta, "index": i}})
        print(f"{name}: {sum(r['set'] == name for r in rows)} of {len(items)} crops", flush=True)
    return rows


def pages() -> list[dict]:
    rows = []
    for name, lang, ps in (("synth", "hi+en", read_set("test_unseen")), ("cord", "en", cord_pages("test"))):
        for p in ps:
            file = f"pages/{name}/{p.id}.png"
            write(OUT / file, p.image())
            h, w = p.image().shape[:2]
            rows.append({"id": f"{name}/{p.id}", "set": name, "file": file, "text": p.text, "lang": lang,
                         "size": [w, h], "meta": p.meta,
                         "lines": [{"quad": line.quad.round(1).tolist(), "text": line.text} for line in p.lines]})
        print(f"{name}: {len(ps)} pages", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=1000, help="crops drawn from each set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    digest = hashlib.sha256()
    for task, rows in (("crops", crops(args.per_set, args.seed)), ("pages", pages())):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (OUT / f"{task}.jsonl").write_text(text, encoding="utf8")
        digest.update(text.encode("utf8"))
        for r in rows:
            digest.update((OUT / r["file"]).read_bytes())
    info = {"per_set": args.per_set, "seed": args.seed, "sha256": digest.hexdigest()}
    (OUT / "export.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    print(f"wrote {OUT}: digest {info['sha256'][:16]}")


if __name__ == "__main__":
    main()
