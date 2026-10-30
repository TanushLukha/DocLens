import cv2
import numpy as np

from doclens.pagereader import pieces
from doclens.pages import Line, Page, box_quad, crop_line, order_quad, quad_size, read_set, write_set
from doclens.tesseract import parse_tsv


def test_order_quad_starts_top_left_and_goes_clockwise():
    assert order_quad(np.float32([[10, 0], [10, 5], [0, 5], [0, 0]])).tolist() == [[0, 0], [10, 0], [10, 5], [0, 5]]
    q = order_quad(cv2.boxPoints(((50, 20), (80, 10), 3)))           # turned by 3 degrees
    assert q[0][0] < q[1][0] and q[3][1] > q[0][1]
    w, h = quad_size(q)
    assert abs(w - 80) < 1 and abs(h - 10) < 1


def test_crop_line_levels_a_tilted_line():
    img = np.full((200, 400), 255, np.uint8)
    cv2.rectangle(img, (100, 90), (299, 109), 0, -1)                   # a dark bar 200 x 20: the line
    m = cv2.getRotationMatrix2D((200, 100), 5, 1.0)
    turned = cv2.warpAffine(img, m, (400, 200), borderValue=255)
    quad = cv2.transform(box_quad(100, 90, 300, 110).reshape(-1, 1, 2), m).reshape(4, 2)
    crop = crop_line(turned, quad, margin=0.5)                         # 10 pixels of page on every side
    assert abs(crop.shape[1] - 220) <= 2 and abs(crop.shape[0] - 40) <= 2
    assert crop[12:-12, 12:-12].mean() < 60                             # the bar fills the middle: it came out level
    assert crop[:6].mean() > 200 and crop[-6:].mean() > 200             # and paper above and below it


def test_page_sets_round_trip(tmp_path):
    page = Page("a", [Line(box_quad(1, 2, 30, 12), "पहली पंक्ति"), Line(box_quad(1, 20, 25, 30), "second")],
                {"script": "mixed"})
    write_set(tmp_path / "s", [(np.full((40, 40), 250, np.uint8), page)])
    back = read_set("s", root=tmp_path)[0]
    assert back.text == "पहली पंक्ति\nsecond" and back.meta == {"script": "mixed"}
    assert np.allclose(back.lines[0].quad, page.lines[0].quad) and back.image().shape == (40, 40)


def test_tesseract_lines_come_from_their_words():
    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = ["4\t1\t1\t1\t1\t0\t10\t10\t200\t30\t-1\t",
            "5\t1\t1\t1\t1\t1\t10\t12\t80\t25\t95\tHello",
            "5\t1\t1\t1\t1\t2\t100\t14\t110\t26\t95\tworld",
            "4\t1\t1\t1\t2\t0\t10\t50\t60\t30\t-1\t",
            "5\t1\t1\t1\t2\t1\t10\t50\t60\t30\t91\tनमस्ते",
            "4\t1\t2\t1\t1\t0\t300\t300\t50\t20\t-1\t"]                # a line without words is dropped
    r = parse_tsv("\n".join([head, *rows]))
    assert r.texts == ["Hello world", "नमस्ते"]
    assert abs(r.quads[0].min(axis=0) - [10, 12]).max() < 2 and abs(r.quads[0].max(axis=0) - [210, 40]).max() < 2


def test_long_lines_are_cut_at_word_gaps():
    crop = np.full((50, 2000), 255, np.uint8)
    for x in range(20, 1960, 120):                                     # words 100 wide, gaps of 20
        crop[10:40, x:x + 100] = 0
    parts = pieces(crop, 24)
    assert len(parts) >= 2 and all(p.shape[1] <= 24 * 50 for p in parts) and sum(p.shape[1] for p in parts) == 2000
    assert len(pieces(crop[:, :1000], 24)) == 1
