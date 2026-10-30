"""M3: builds the synthetic page sets: held out text set as pages, with every line's box and text.

    python scripts/make_pages.py --corpus ../TinyGPT/data/clean

test_seen and test_unseen hold the same 30 pages of text (Hindi, English and mixed; one and two
columns), set with the training fonts and with the six held out fonts, each page clean and under
the four page damages: 150 pages per set. val holds 6 more pages of other text, training fonts,
for tuning the classical line finder. Hindi text comes from Hindi Wikipedia documents in TinyGPT's
validation split, English from the held out English sentences; validation and test pages never
share a document or a sentence. Everything is seeded: the same command gives the same pixels.
"""
import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, fonts  # noqa: E402
from doclens.corpus import english_sentences, sentences, tinygpt_docs  # noqa: E402
from doclens.pagegen import CONDITIONS, compose, damage, draw_style  # noqa: E402
from doclens.pages import PAGES, Page, write_set  # noqa: E402
from doclens.text import script_of  # noqa: E402

SCRIPTS = ("deva", "latn", "mixed")


def part(key: str, val_share: float) -> str:
    """val or test, by a hash of the text: stable whatever the order of the sources. MD5, not the SHA1
    that chose the held out English sentences in the first place (every one of those has a small
    SHA1 value, so SHA1 again would put them all on one side)."""
    return "val" if int(hashlib.md5(key.encode("utf8")).hexdigest()[:8], 16) % 1000 < 1000 * val_share else "test"


def hindi_docs(clean_dir: Path) -> dict[str, list[list[str]]]:
    """Hindi Wikipedia documents of TinyGPT's validation split, as lists of sentences, by part."""
    out: dict[str, list[list[str]]] = {"val": [], "test": []}
    for text, src in tinygpt_docs(clean_dir, "val"):
        if src != "wiki_hi":
            continue
        sents = [s for s in sentences(text) if script_of(s) in ("deva", "mixed") and len(s) >= 20]
        if len(sents) >= 5:
            out[part(text[:200], 0.15)].append(sents)
    return out


def paragraphs(rng: random.Random, script: str, hindi: list[list[str]], english: list[str]) -> Iterator[str]:
    """An endless stream of paragraphs: Hindi ones keep a document's sentences in order, English ones
    join held out sentences; mixed pages alternate the two."""
    turn = 0
    while True:
        use = script if script != "mixed" else ("deva" if turn % 2 == 0 else "latn")
        turn += 1
        k = rng.randint(2, 5)
        if use == "deva":
            doc = hindi[rng.randrange(len(hindi))]
            i = rng.randrange(max(1, len(doc) - k))
            yield " ".join(doc[i:i + k])
        else:
            yield " ".join(rng.sample(english, k))


def title_of(rng: random.Random, script: str, hindi: list[list[str]], english: list[str]) -> str:
    src = rng.choice(hindi)[0] if script in ("deva", "mixed") else rng.choice(english)
    return " ".join(src.split()[: rng.randint(3, 7)]).rstrip(",:;")


def build(name: str, bases: list[tuple[str, int, int]], split: str, hindi: list[list[str]], english: list[str],
          seed: int) -> None:
    deva_fonts, latn_fonts = fonts.load(split, "deva"), fonts.load(split, "latn")
    out = []
    for b, (script, columns, k) in enumerate(bases):
        # the text and the page style depend on the base page only, so seen and unseen sets share them
        trng = random.Random(f"{seed}/{script}/{columns}/{k}")
        rng = np.random.default_rng([seed, b])
        style = draw_style(rng, columns)
        deva = deva_fonts[int(rng.integers(len(deva_fonts)))]
        latn = latn_fonts[int(rng.integers(len(latn_fonts)))]
        img, lines = compose(title_of(trng, script, hindi, english), paragraphs(trng, script, hindi, english),
                             int(trng.randint(2, 300)), deva, latn, style)
        for ci, cond in enumerate(CONDITIONS):
            dimg, dlines = damage(img, lines, cond, np.random.default_rng([seed, b, ci]))
            pid = f"{script}_{columns}col_{k:02d}_{cond}"
            meta = {"set": name, "script": script, "columns": columns, "condition": cond,
                    "base": f"{script}_{columns}col_{k:02d}", "fonts": [deva.name, latn.name], "size": style.size}
            out.append((dimg, Page(pid, dlines, meta)))
    write_set(PAGES / name, out)
    n = sum(len(p.lines) for _, p in out)
    print(f"{name}: {len(out)} pages, {n} lines, fonts {sorted({f for _, p in out for f in p.meta['fonts']})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../TinyGPT/data/clean", help="TinyGPT's data/clean folder")
    ap.add_argument("--english", default=str(DATA / "text" / "english.txt"))
    ap.add_argument("--per-layout", type=int, default=5, help="test pages per script and column count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fonts.require_shaping()
    docs = hindi_docs(Path(args.corpus))
    eng = list(english_sentences(Path(args.english), "val"))
    english = {p: [s for s in eng if part(s, 0.2) == p and len(s) >= 20] for p in ("val", "test")}
    print(f"text: Hindi documents {len(docs['val'])} val, {len(docs['test'])} test; English sentences"
          f" {len(english['val'])} val, {len(english['test'])} test")
    tests = [(s, c, k) for s in SCRIPTS for c in (1, 2) for k in range(args.per_layout)]
    vals = [(s, c, 100) for s in SCRIPTS for c in (1, 2)]
    build("val", vals, "train", docs["val"], english["val"], args.seed + 1)
    build("test_seen", tests, "train", docs["test"], english["test"], args.seed)
    build("test_unseen", tests, "test", docs["test"], english["test"], args.seed)
    (PAGES / "README.json").write_text(json.dumps({"sets": ["val", "test_seen", "test_unseen"], "seed": args.seed,
                                                   "per_layout": args.per_layout}), encoding="utf8")


if __name__ == "__main__":
    main()
