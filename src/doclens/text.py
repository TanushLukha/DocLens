"""Text normalisation shared by every dataset, OCR engine and metric.

The rule is the same as TinyGPT's: Unicode NFC, the two joiners kept (they change how a
Devanagari conjunct is drawn), other invisible characters dropped, whitespace collapsed.
Ground truth and OCR output both pass through `normalize` before any score is computed,
so a model is never charged for a difference nobody can see.
"""
from __future__ import annotations

import unicodedata

import regex

ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
KEEP_FORMAT = {ZWNJ, ZWJ}
DANDA, DOUBLE_DANDA = chr(0x0964), chr(0x0965)

_DEVANAGARI = regex.compile(r"[[\p{L}\p{M}]&&\p{Script=Devanagari}]", regex.V1)
_LATIN = regex.compile(r"[[\p{L}\p{M}]&&\p{Script=Latin}]", regex.V1)
_LETTER = regex.compile(r"[\p{L}\p{M}]")


def normalize(text: str) -> str:
    """NFC, drop invisible characters except ZWNJ and ZWJ, one space between words, no line breaks."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch in KEEP_FORMAT or not unicodedata.category(ch).startswith("C")
                   or ch in "\n\t")
    return " ".join(text.split())


def is_devanagari(ch: str) -> bool:
    """Devanagari block, plus the two joiners, which only ever sit inside Devanagari words here."""
    return 0x0900 <= ord(ch) <= 0x097F or 0xA8E0 <= ord(ch) <= 0xA8FF or ch in KEEP_FORMAT


def script_shares(text: str) -> tuple[float, float]:
    """(Devanagari share, Latin share) among the letters and marks of `text`."""
    letters = len(_LETTER.findall(text))
    if letters == 0:
        return 0.0, 0.0
    return len(_DEVANAGARI.findall(text)) / letters, len(_LATIN.findall(text)) / letters


def wrong_script(ref: str, hyp: str) -> bool:
    """True when the output has letters of a script the reference has none of.

    Counted only when the reference has letters of exactly one script: a Hindi word read
    as Latin letters, or an English word read as Devanagari. This is what a two language
    setting (-l hin+eng) does to short crops, where there is no context to pick the script.
    Digits and punctuation are not letters, so a number never counts.
    """
    rd, rl = bool(_DEVANAGARI.search(ref)), bool(_LATIN.search(ref))
    if rd == rl:
        return False
    hd, hl = bool(_DEVANAGARI.search(hyp)), bool(_LATIN.search(hyp))
    return (rd and hl) or (rl and hd)


def script_of(text: str) -> str:
    """deva, latn or mixed: the label a text line gets in every results table."""
    d, l = script_shares(text)
    if d >= 0.95:
        return "deva"
    if l >= 0.95:
        return "latn"
    return "mixed"
