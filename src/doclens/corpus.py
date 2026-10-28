"""Text to draw: printed lines cut from TinyGPT's cleaned corpus and from English sentences.

Hindi lines come from Hindi Wikipedia, Latin lines from English sentences (or, when those
are missing, from TinyGPT's Hinglish), and mixed lines are the Wikipedia sentences that
already mix the two scripts, as in "हाइड्रोजन (अंग्रेज़ी: Hydrogen) एक रासायनिक तत्व है".
Test lines come only from TinyGPT's validation documents, which TinyGPT never trained on,
so in M2 its language model can rescore OCR output without having memorised the answer.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterator

from .text import normalize, script_of

_SENTENCE_END = re.compile(r"(?<=[" + chr(0x0964) + chr(0x0965) + r".!?])\s+")


def sentences(text: str) -> Iterator[str]:
    for para in text.split("\n"):
        for s in _SENTENCE_END.split(para):
            s = normalize(s)
            if s:
                yield s


def to_lines(sentence: str, rng: random.Random, min_len: int = 25, max_len: int = 70) -> Iterator[str]:
    """Pack words greedily into lines whose target length is drawn between min_len and max_len."""
    words, cur, target = sentence.split(), [], rng.randint(min_len, max_len)
    for w in words:
        if cur and len(" ".join(cur + [w])) > target:
            yield " ".join(cur)
            cur, target = [], rng.randint(min_len, max_len)
        cur.append(w)
    if cur and len(" ".join(cur)) >= min_len // 2:
        yield " ".join(cur)


def tinygpt_docs(clean_dir: Path, split: str) -> Iterator[tuple[str, str]]:
    """(text, source) from TinyGPT's data/clean/<split>.jsonl."""
    with open(Path(clean_dir) / f"{split}.jsonl", encoding="utf8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec["text"], rec.get("src", "")


def english_sentences(path: Path, split: str) -> Iterator[str]:
    """English sentences from data/text/english.txt, split 99.5 / 0.5 by a hash of the sentence."""
    with open(path, encoding="utf8") as f:
        for line in f:
            s = normalize(line)
            if not s:
                continue
            h = int(hashlib.sha1(s.encode("utf8")).hexdigest()[:8], 16) % 1000
            if (h < 5) == (split == "val"):
                yield s


def collect_lines(clean_dir: Path, split: str, english: Path | None, seed: int = 0) -> dict[str, list[str]]:
    """All candidate lines of the split, grouped by script: deva, latn, mixed. Duplicates removed."""
    rng = random.Random(seed)
    out: dict[str, list[str]] = {"deva": [], "latn": [], "mixed": []}
    seen: set[str] = set()

    def add(line: str, want: str | None = None) -> None:
        k = script_of(line)
        if (want is None or k == want) and line not in seen:
            seen.add(line)
            out[k].append(line)

    for text, src in tinygpt_docs(clean_dir, split):
        hindi = src == "wiki_hi" or script_of(text) != "latn"
        if not hindi and english is not None:
            continue                       # Latin lines come from English when it is there
        for s in sentences(text):
            for line in to_lines(s, rng):
                add(line)
    if english is not None and Path(english).exists():
        for s in english_sentences(Path(english), split):
            for line in to_lines(s, rng):
                add(line, "latn")
    return out
