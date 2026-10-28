import shutil

import numpy as np
import pytest

from doclens import DATA, fonts, tesseract
from doclens.render import render_line
from doclens.tesseract import TessConfig

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def test_batched_results_stay_in_order_even_with_a_blank_image():
    f = next(f for f in fonts.load(script="latn") if f.name == "Liberation Serif")
    a = render_line("The quick brown fox jumps over the lazy dog.", f, f, 32).image
    b = render_line("Invoice 4521, total 1,250.00", f, f, 32).image
    blank = np.full((60, 300), 245, np.uint8)
    out = tesseract.recognize([a, blank, b] * 7, TessConfig(langs="eng"), workers=2, chunk=4)
    assert len(out) == 21
    assert out[0] == "The quick brown fox jumps over the lazy dog."
    assert out[2] == "Invoice 4521, total 1,250.00"
    assert out[18] == out[0] and out[20] == out[2]


def test_hindi_line_is_read():
    folder = DATA / "tessdata_fast"          # downloaded by get_data.py; Homebrew ships no Hindi model
    tessdata = str(folder) if folder.exists() else None
    if "hin" not in tesseract.languages(tessdata):
        pytest.skip("no Hindi model yet: run scripts/get_data.py --only tessdata")
    d = next(f for f in fonts.load(script="deva") if f.name == "Noto Sans Devanagari")
    img = render_line("भारत एक विशाल देश है", d, d, 36).image
    out = tesseract.recognize([img], TessConfig(langs="hin", tessdata=tessdata))
    assert out[0] == "भारत एक विशाल देश है"
