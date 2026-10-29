"""Prepares everything M2's training reads, into data/m2/ and data/synth/val/ (never committed).

    python scripts/prepare_m2.py --corpus ../TinyGPT/data/clean

1. Training text: lines cut from TinyGPT's training documents and English sentences (the 99.5%
   not held out), grouped by script, capped per script, into data/m2/train_lines.tsv.
2. Synthetic validation lines: held out text (TinyGPT's validation documents, English held out
   sentences) minus every line of the two test sets, drawn with the training fonts, each under one
   random condition, into data/synth/val/. For choosing checkpoints and decoding settings, never the
   held out fonts.
3. CORD crops: words and lines of the training and validation receipts as PNG files,
   into data/m2/cord_train/ and data/m2/cord_validation/ (needs get_data.py --only cord_train).
4. The alphabet: every character of the real training text plus the frequent, drawable characters
   of the synthetic text, into data/m2/alphabet.json, with how much of each validation set it covers.

Everything is seeded: the same command gives the same files.
"""
import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, alphabet, datasets, fonts  # noqa: E402
from doclens.corpus import collect_lines  # noqa: E402
from doclens.degrade import CONDITIONS  # noqa: E402
from doclens.render import render_line  # noqa: E402
from doclens.text import normalize  # noqa: E402
from doclens.traindata import M2, read_lines  # noqa: E402

CAPS = {"deva": 300_000, "latn": 150_000, "mixed": 100_000}
VAL = {"deva": 600, "latn": 300, "mixed": 100}     # mixed: only a few held out lines are not in the test sets


def training_text(corpus: Path, english: Path | None, seed: int) -> dict[str, list[str]]:
    t0 = time.time()
    pool = collect_lines(corpus, "train", english, seed)
    rng = random.Random(seed)
    out = {k: rng.sample(v, min(len(v), CAPS[k])) for k, v in pool.items()}
    rows = [[k, t] for k, v in out.items() for t in v]
    M2.mkdir(parents=True, exist_ok=True)
    datasets.write_tsv(M2 / "train_lines.tsv", ["script", "text"], rows)
    print(f"training text: {({k: len(v) for k, v in pool.items()})} candidate lines, kept"
          f" {({k: len(v) for k, v in out.items()})} ({time.time() - t0:.0f} s)")
    return out


def synthetic_val(corpus: Path, english: Path | None, seed: int) -> None:
    out = DATA / "synth" / "val"
    if (out / "labels.tsv").exists():
        print(f"have {out.relative_to(DATA)}")
        return
    test_lines = {it.text for name in ("test_seen", "test_unseen") if (DATA / "synth" / name).exists()
                  for it in datasets.synth(name)}
    pool = collect_lines(corpus, "val", english, seed)
    rng = random.Random(seed + 7)
    chosen = []
    for script, n in VAL.items():
        cands = [t for t in pool[script] if t not in test_lines]
        chosen += [(script, t) for t in rng.sample(cands, min(n, len(cands)))]
    deva_fonts, latn_fonts = fonts.load("train", "deva"), fonts.load("train", "latn")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, conds = [], list(CONDITIONS)
    for i, (script, text) in enumerate(chosen):
        g = np.random.default_rng([seed, 99, i])
        deva = deva_fonts[int(g.integers(len(deva_fonts)))]
        latn = latn_fonts[int(g.integers(len(latn_fonts)))]
        size = int(g.integers(24, 41))
        try:
            r = render_line(text, deva, latn, size, latin_in_deva_font=bool(g.random() < 0.5),
                            ink=int(g.integers(0, 60)), paper=int(g.integers(205, 256)))
        except ValueError:
            continue
        cond = conds[i % len(conds)]                       # every condition equally often
        img = CONDITIONS[cond](r.image, g)
        file = f"images/{script}_{i:05d}_{cond}.png"
        cv2.imwrite(str(out / file), img)
        rows.append({"file": file, "text": text, "condition": cond, "fonts": "+".join(r.fonts),
                     "script": script, "size": size})
    datasets.write_synth(out, rows)
    overlap = sum(r["text"] in test_lines for r in rows)
    print(f"synthetic validation: {len(rows)} images, {overlap} lines shared with the test sets")


def cord_crops() -> None:
    for split in ("train", "validation"):
        folder = M2 / f"cord_{split}"
        if (folder / "labels.tsv").exists():
            print(f"have {folder.relative_to(DATA)}")
            continue
        try:
            items = []
            for level in ("word", "line"):
                for it in datasets.cord(split, level):
                    it.meta["level"] = level
                    items.append(it)
        except FileNotFoundError as e:
            print(f"SKIP CORD {split}: {e} (python scripts/get_data.py --only cord_train)")
            continue
        datasets.write_crops(folder, items)
        print(f"CORD {split}: {sum(i.meta['level'] == 'word' for i in items)} words,"
              f" {sum(i.meta['level'] == 'line' for i in items)} lines")


def build_alphabet(lines: dict[str, list[str]]) -> None:
    synthetic = Counter(ch for v in lines.values() for t in v for ch in t)
    real = Counter()
    for it in datasets.mozhi("train"):
        real.update(it.text)
    cord_train = M2 / "cord_train"
    if (cord_train / "labels.tsv").exists():
        for it in datasets.crops(cord_train):
            real.update(it.text)
    train_fonts = fonts.load("train")
    abc = alphabet.build(synthetic, real, lambda ch: any(f.covers(ch) for f in train_fonts))
    abc.save(M2 / "alphabet.json")
    print(f"alphabet: {len(abc.chars)} characters (+ the blank)")
    report = {}
    try:
        report["mozhi val"] = [it.text for it in datasets.mozhi("val")]
    except (FileNotFoundError, ValueError):
        pass
    if (DATA / "synth" / "val").exists():
        report["synthetic val"] = [it.text for it in datasets.synth("val")]
    if (M2 / "cord_validation" / "labels.tsv").exists():
        report["CORD val"] = [it.text for it in datasets.crops(M2 / "cord_validation")]
    for name, texts in report.items():
        chars = sum(len(t) for t in texts)
        missing = Counter(ch for t in texts for ch in abc.missing(normalize(t)))
        print(f"     {name}: {100 * (1 - sum(missing.values()) / max(chars, 1)):.3f}% of characters covered;"
              f" missing: {dict(missing.most_common(8))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../TinyGPT/data/clean", help="TinyGPT's data/clean folder")
    ap.add_argument("--english", default=str(DATA / "text" / "english.txt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    corpus = Path(args.corpus)
    if not (corpus / "train.jsonl").exists():
        sys.exit(f"no {corpus / 'train.jsonl'}: point --corpus at TinyGPT's data/clean folder")
    english = Path(args.english) if Path(args.english).exists() else None
    if (M2 / "train_lines.tsv").exists():
        lines = read_lines(M2 / "train_lines.tsv")
        print(f"have data/m2/train_lines.tsv: {({k: len(v) for k, v in lines.items()})}")
    else:
        lines = training_text(corpus, english, args.seed)
    synthetic_val(corpus, english, args.seed)
    cord_crops()
    build_alphabet(lines)
