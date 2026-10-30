import numpy as np
import pytest

from doclens import fonts
from doclens.layout import LineFinder, Params
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


def test_a_paragraph_break_across_both_columns_is_not_taken_for_a_title():
    # Word boxes as the finder sees them (text 15 pixels high, lines 32 apart): a title, two columns
    # whose lines are offset by half a pitch, so the only band of white across both columns below
    # the title is a paragraph break low on the page, one long line near the top that narrows the
    # gap between the columns, and a page number in that gap. The part above the break is far too
    # tall to be a title: each column must come out whole, the left one first. Gaps are searched
    # upright only, so that this tests the rule for titles and page numbers alone.
    boxes, labels = [(450, 100, 650, 130)], ["title"]
    for side, x0, dy in (("left", 100, 0), ("right", 580, 16)):
        for i in range(30):
            y = 200 + 32 * i + dy + (40 if i >= 25 else 0)
            words = [(x0, x0 + 100), (x0 + 110, x0 + 230), (x0 + 240, x0 + 380)]
            if (side, i) == ("left", 5):
                words.append((490, 520))                              # the long line
            boxes += [(a, y, b, y + 15) for a, b in words]
            labels += [side] * len(words)
    boxes.append((545, 1500, 555, 1515))
    labels.append("page number")
    f = LineFinder(Params(slant=0.0))
    f.boxes, f.h, f.pitch = np.array(boxes, float), 15.0, 32.0
    order = [labels[i] for block in f._xy_cut(np.arange(len(boxes))) for i in block]
    assert order == ["title"] + ["left"] * labels.count("left") + ["right"] * labels.count("right") + ["page number"]


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
