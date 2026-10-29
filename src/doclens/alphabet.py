"""The characters the recognizer can write, and the numbers it uses for them.

Index 0 is CTC's blank, the label for "no new character here"; every other index is one Unicode
code point after `text.normalize`, so the recognizer is scored exactly as Tesseract was. The
alphabet is built from training data only. A test character it lacks can never be read, and the
evaluation reports how often that happens instead of quietly adding test characters.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from .text import KEEP_FORMAT

BLANK = 0


class Alphabet:
    def __init__(self, chars: Iterable[str]):
        chars = list(chars)
        if any(len(c) != 1 for c in chars):
            raise ValueError("every symbol must be one code point")
        if len(set(chars)) != len(chars):
            raise ValueError("a character appears twice")
        self.chars = chars
        self.index = {c: i + 1 for i, c in enumerate(chars)}

    def __len__(self) -> int:
        """Number of output classes, the blank included."""
        return len(self.chars) + 1

    def covers(self, text: str) -> bool:
        return all(ch in self.index for ch in text)

    def missing(self, text: str) -> list[str]:
        return [ch for ch in text if ch not in self.index]

    def encode(self, text: str) -> list[int]:
        """Class indices of a normalised text; KeyError for a character outside the alphabet."""
        return [self.index[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.chars[i - 1] for i in ids if i != BLANK)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": "".join(self.chars)}, ensure_ascii=False) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path: str | Path) -> Alphabet:
        return cls(json.loads(Path(path).read_text(encoding="utf8"))["chars"])


def printable(ch: str) -> bool:
    """Letters, marks, digits, punctuation, symbols, the space and the two joiners; no control characters."""
    return ch == " " or ch in KEEP_FORMAT or not unicodedata.category(ch).startswith(("C", "Z"))


def build(synthetic: Counter, real: Counter, drawable: Callable[[str], bool], min_count: int = 50) -> Alphabet:
    """Every character of the real training text (Mozhi, CORD), plus the characters of the synthetic
    text that occur at least `min_count` times and that a training font can draw. Sorted by code point.

    The threshold drops the long tail of Wikipedia (a stray Chinese or Arabic letter) that the
    recognizer would see too rarely to learn, and that no test set asks for.
    """
    keep = {c for c in real if printable(c)}
    keep |= {c for c, n in synthetic.items() if n >= min_count and printable(c) and drawable(c)}
    return Alphabet(sorted(keep))
