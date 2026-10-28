import numpy as np
import pytest

from doclens import fonts
from doclens.render import render_line, runs

pytestmark = pytest.mark.skipif(not __import__("PIL.features").features.check("raqm"),
                                reason="Pillow without libraqm cannot shape Devanagari")


def deva(name="Noto Sans Devanagari"):
    return next(f for f in fonts.load(script="deva") if f.name == name)


def latn(name="Liberation Sans"):
    return next(f for f in fonts.load(script="latn") if f.name == name)


def test_every_font_file_exists_and_splits_are_disjoint():
    all_fonts = fonts.load()
    assert all(f.path.exists() for f in all_fonts)
    assert {f.file for f in fonts.load("train")}.isdisjoint({f.file for f in fonts.load("test")})
    for script in ("deva", "latn"):
        assert fonts.load("train", script) and fonts.load("test", script)


def test_every_devanagari_font_shapes_conjuncts():
    for f in fonts.load(script="deva"):
        assert fonts.shaping_works(f), f.name


def test_runs_send_latin_to_the_latin_font_when_the_devanagari_font_lacks_it():
    # Noto Sans Devanagari has no Latin letters, so "IIT" must go to the Latin font
    parts = runs("हिन्दी IIT 1250", deva(), latn(), latin_in_deva_font=True)
    assert "".join(s for s, _ in parts) == "हिन्दी IIT 1250"
    assert any("IIT" in s and f.name == "Liberation Sans" for s, f in parts)


def test_render_is_deterministic_and_has_ink():
    a = render_line("कृष्ण और Krishna", deva(), latn(), 32)
    b = render_line("कृष्ण और Krishna", deva(), latn(), 32)
    assert np.array_equal(a.image, b.image)
    assert a.image.min() < 100 < a.image.max()
    assert a.image.shape[0] < a.image.shape[1]
