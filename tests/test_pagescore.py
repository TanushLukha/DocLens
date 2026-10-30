import pytest

from doclens.pages import Line, box_quad
from doclens.pagescore import iou_matrix, kendall, match, page_edits, score_page, splits_merges


def test_iou_and_one_to_one_matching():
    true = [box_quad(0, 0, 100, 10), box_quad(0, 20, 100, 30)]
    found = [box_quad(0, 0, 100, 10), box_quad(0, 1, 100, 11), box_quad(0, 19, 100, 31)]
    iou = iou_matrix(true, found)
    assert iou[0, 0] == pytest.approx(1.0) and iou[1, 0] == 0
    assert match(iou) == [(0, 0), (1, 2)]                  # the second box over line 0 stays unmatched


def test_splits_and_merges():
    true = [box_quad(0, 0, 100, 10), box_quad(0, 20, 100, 30)]
    assert splits_merges(true, [box_quad(0, 0, 45, 10), box_quad(55, 0, 100, 10)]) == (1, 0)
    assert splits_merges(true, [box_quad(0, 0, 100, 30)]) == (0, 1)


def test_kendall_tau():
    assert kendall([(0, 0), (1, 1), (2, 2)]) == 1.0
    assert kendall([(0, 2), (1, 1), (2, 0)]) == -1.0
    assert kendall([(0, 0)]) is None


def test_page_text_ignores_line_breaks_and_separator_runs():
    assert page_edits("a b\nc", "a\nb c") == (0, 5)                # a line cut in two costs nothing
    assert page_edits("TOTAL 5", "-------\nTOTAL 5\n=====") == (0, 7)
    assert page_edits("ab\ncd", "cd\nab")[0] > 0                   # the wrong order does


def test_score_page():
    lines = [Line(box_quad(0, 0, 100, 10), "ab"), Line(box_quad(0, 20, 100, 30), "cd")]
    s = score_page(lines, [box_quad(0, 20, 100, 30), box_quad(0, 0, 100, 10)], "cd\nab")
    assert s["matched"] == 2 and s["tau"] == -1.0 and s["edits"] > 0 and s["chars"] == 5
    assert "matched" not in score_page(lines, None, "ab cd") and score_page(lines, None, "ab cd")["edits"] == 0
