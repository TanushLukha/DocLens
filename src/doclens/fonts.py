"""The fonts DocLens renders text with, kept in the repository so every machine draws the same pixels.

fonts/fonts.json lists each file with the scripts it covers, its licence and its split:
"train" fonts render training images, "test" fonts are held out, so a recognizer can be
measured on letter shapes it has never seen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PIL import ImageFont, features

FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"


@dataclass(frozen=True)
class Font:
    file: str
    name: str
    scripts: tuple[str, ...]
    split: str
    license: str

    @property
    def path(self) -> Path:
        return FONT_DIR / self.file

    def covers(self, ch: str) -> bool:
        return ch in (" ", chr(0x200C), chr(0x200D)) or ord(ch) in _cmap(str(self.path))

    def pil(self, size: int) -> ImageFont.FreeTypeFont:
        return _pil(str(self.path), size)


@cache
def _cmap(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont
    return frozenset(TTFont(path, lazy=True).getBestCmap())


@cache
def _pil(path: str, size: int) -> ImageFont.FreeTypeFont:
    require_shaping()
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def load(split: str | None = None, script: str | None = None) -> list[Font]:
    rows = json.loads((FONT_DIR / "fonts.json").read_text(encoding="utf8"))
    fonts = [Font(r["file"], r["name"], tuple(r["scripts"]), r["split"], r["license"]) for r in rows]
    return [f for f in fonts if (split is None or f.split == split) and (script is None or script in f.scripts)]


def require_shaping() -> None:
    """Devanagari must be shaped (conjuncts, vowel signs, reph); Pillow does it only through libraqm.

    Without it Pillow silently falls back to drawing one glyph per code point, which makes
    broken text that still looks like Hindi to an untrained eye. Refuse instead.
    """
    if not features.check("raqm"):
        raise RuntimeError("Pillow cannot shape Devanagari: its libraqm needs FriBiDi. On a Mac: "
                           "brew install fribidi, and if that is not enough, "
                           "export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib (add it to ~/.zshrc). "
                           "Check with: python scripts/check_env.py")


def shaping_works(font: Font, size: int = 40) -> bool:
    """A shaped conjunct is narrower than its three code points drawn side by side."""
    text = chr(0x0915) + chr(0x094D) + chr(0x0937)          # क्ष
    shaped = font.pil(size).getlength(text)
    basic = ImageFont.truetype(str(font.path), size, layout_engine=ImageFont.Layout.BASIC).getlength(text)
    return shaped < basic
