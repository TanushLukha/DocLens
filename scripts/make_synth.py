"""Builds the fixed synthetic test sets: held out text lines, rendered, then damaged.

    python scripts/make_synth.py --corpus ../TinyGPT/data/clean

Two sets with the same recipe: test_seen draws with the training fonts, test_unseen with
the held out ones (M2's recognizer never sees them). Every line is saved clean and under
each damage of degrade.CONDITIONS, with the same text and font, so conditions compare
like for like. Everything is seeded: the same command gives the same pixels.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, fonts  # noqa: E402
from doclens.corpus import collect_lines  # noqa: E402
from doclens.datasets import write_synth  # noqa: E402
from doclens.degrade import CONDITIONS  # noqa: E402
from doclens.render import render_line  # noqa: E402


def build(name: str, lines: list[tuple[str, str]], split: str, seed: int) -> None:
    out = DATA / "synth" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    deva_fonts, latn_fonts = fonts.load(split, "deva"), fonts.load(split, "latn")
    rows, skipped, used = [], 0, Counter()
    for i, (script, text) in enumerate(lines):
        rng = np.random.default_rng([seed, i])
        deva = deva_fonts[int(rng.integers(len(deva_fonts)))]
        latn = latn_fonts[int(rng.integers(len(latn_fonts)))]
        size = int(rng.integers(24, 41))
        try:
            r = render_line(text, deva, latn, size, latin_in_deva_font=bool(rng.random() < 0.5),
                            ink=int(rng.integers(0, 60)), paper=int(rng.integers(205, 256)))
        except ValueError:
            skipped += 1
            continue
        used.update(r.fonts)
        for ci, (cond, fn) in enumerate(CONDITIONS.items()):
            img = fn(r.image, np.random.default_rng([seed, i, ci]))
            file = f"images/{script}_{i:05d}_{cond}.png"
            cv2.imwrite(str(out / file), img)
            rows.append({"file": file, "text": text, "condition": cond, "fonts": "+".join(r.fonts),
                         "script": script, "size": size})
    write_synth(out, rows)
    print(f"{name}: {len(rows)} images from {len(lines) - skipped} lines ({skipped} skipped: a character no font has)")
    print(f"     fonts used: {dict(used)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../TinyGPT/data/clean", help="TinyGPT's data/clean folder")
    ap.add_argument("--english", default=str(DATA / "text" / "english.txt"))
    ap.add_argument("--deva", type=int, default=400)
    ap.add_argument("--latn", type=int, default=200)
    ap.add_argument("--mixed", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    english = Path(args.english) if Path(args.english).exists() else None
    if english is None:
        print("no English sentences found: Latin lines come from TinyGPT's Hinglish instead")
    pool = collect_lines(Path(args.corpus), "val", english, args.seed)
    print({k: len(v) for k, v in pool.items()}, "candidate lines in held out text")
    rng = random.Random(args.seed)
    chosen = []
    for script, n in (("deva", args.deva), ("latn", args.latn), ("mixed", args.mixed)):
        picks = rng.sample(pool[script], min(n, len(pool[script])))
        chosen += [(script, t) for t in picks]
    build("test_seen", chosen, "train", args.seed)
    build("test_unseen", chosen, "test", args.seed + 1)
