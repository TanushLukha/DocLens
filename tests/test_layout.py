import pytest

from doclens import fonts
from doclens.layout import LineFinder
from doclens.pagegen import Canvas, Style, compose, damage
from doclens.pagescore import score_page

WORDS = ("the quick brown fox jumps over a lazy dog while seven red kites drift above quiet "
         "fields and an old mill turns slowly in the evening wind").split()


def style(columns: int) -> Style:
    return Style(size=44, pitch=70, margins=(200, 220, 200, 260), columns=columns, gap=110, para_gap=30, indent=0,
                 title_size=72, title_centred=True, ink=20, paper=245, latin_in_deva_font=False)


def latin():
    found = fonts.load("train", "latn")
    return next((f for f in found if f.name == "Liberation Serif"), found[0])


def paragraphs():
    k = 0
    while True:
        yield " ".join(WORDS[(k + i) % len(WORDS)] for i in range(45))
        k += 7


@pytest.mark.parametrize("columns,condition", [(1, "clean"), (2, "clean"), (2, "skew")])
def test_lines_and_their_order_are_found_on_a_page(columns, condition):
    import numpy as np
    f = latin()
    img, lines = compose("A short title", paragraphs(), 7, f, f, style(columns))
    img, lines = damage(img, lines, condition, np.random.default_rng(0))
    s = score_page(lines, LineFinder()(img), "")
    assert s["matched"] >= 0.95 * len(lines) and s["merges"] == 0
    assert s["tau"] == 1.0                                     # title, the columns in turn, the page number


def test_a_receipt_row_stays_one_line():
    f = latin()
    c = Canvas(style(1), f, f)
    items = [("1 Hot Tuna", "67,000"), ("2 Iced Tea", "24,000"), ("1 Grilled Baby Potato", "50,500"),
             ("Subtotal", "141,500"), ("Service 5%", "7,075"), ("Total", "148,575")]
    for i, (name, price) in enumerate(items):
        c.put(name, 300, 600 + i * 70, 44)
        c.put(price, 2000 - c.measure(44)(price), 600 + i * 70, 44)
    found = LineFinder()(c.img)
    assert len(found) == len(items)
    assert all(q[:, 0].min() < 320 and q[:, 0].max() > 1950 for q in found)   # each spans name and price
