import numpy as np

from doclens import stats
from doclens.fields import Receipt, bio, counts, decode, f1, field_text, label_list, reading_order
from doclens.pages import box_quad


def test_reading_order_is_rows_top_to_bottom_then_left_to_right():
    quads = [box_quad(300, 100, 360, 120), box_quad(10, 100, 80, 120), box_quad(10, 10, 80, 30), box_quad(200, 12, 260, 30)]
    rows = [7, 7, 3, 3]
    assert reading_order(rows, quads) == [2, 3, 1, 0]


def test_bio_tags_follow_the_order_given():
    assert bio([0, 0, 1, -1, 1], ["a", "a", "b", "x", "b"]) == ["B-a", "I-a", "B-b", "O", "I-b"]


def test_decode_continues_a_field_across_other_words():
    words = ["1", "Hot", "67,000", "Tuna"]
    tags = ["B-menu.cnt", "B-menu.nm", "B-menu.price", "I-menu.nm"]          # the name wraps to the next row
    assert decode(words, tags) == [("menu.cnt", "1"), ("menu.nm", "Hot Tuna"), ("menu.price", "67,000")]
    assert decode(["x", "y"], ["I-total", "O"]) == [("total", "x")]          # an I with nothing open starts a field


def test_field_counts_ignore_spaces_and_count_repeats():
    true = [("menu.nm", "HOT TUNA"), ("menu.nm", "HOT TUNA"), ("total", "148,127")]
    pred = [("menu.nm", "HOTTUNA"), ("total", "148.127"), ("menu.cnt", "1")]
    assert counts(true, pred) == (1, 2, 2)
    assert f1(1, 2, 2) == 2 / 6


def test_true_fields_come_from_groups():
    r = Receipt("0", ["Sub", "Total", "36,500", "1"], [box_quad(0, 0, 1, 1)] * 4, (100, 100),
                ["B-sub", "I-sub", "I-sub", "B-cnt"], [4, 4, 4, 9])
    assert r.fields() == [("sub", "Sub Total 36,500"), ("cnt", "1")]
    assert label_list([r]) == ["O", "B-cnt", "I-cnt", "B-sub", "I-sub"]
    assert field_text(["a ", " b"]) == "a b"


def test_f1_intervals_and_paired_differences():
    rng = np.random.default_rng(0)
    tp = rng.integers(5, 10, 50)
    fp, fn = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    r = stats.bootstrap_f1(tp, fp, fn)
    assert r["lo"] <= r["f1"] <= r["hi"]
    same = stats.paired_f1((tp, fp, fn), (tp, fp, fn))
    assert same["diff"] == 0 and same["p"] == 1.0
    better = stats.paired_f1((tp + 3, fp, fn), (tp, fp, fn))
    assert better["diff"] > 0 and better["lo"] > 0 and better["p"] < 0.01


def test_ocr_words_take_the_field_of_the_true_word_under_them():
    from doclens.fields import ocr_receipt, transfer
    quads = [box_quad(0, 0, 50, 20), box_quad(60, 0, 110, 20), box_quad(300, 0, 380, 20)]
    truth = Receipt("0", ["Grand", "Total", "40,150"], quads, (400, 100), ["B-total", "I-total", "I-total"], [7, 7, 7])
    read = [box_quad(0, 0, 110, 20), box_quad(300, 0, 381, 21), box_quad(200, 60, 250, 80), box_quad(0, 0, 1, 1)]
    ocr = ocr_receipt("0", read, ["GrandTotal", "40,150", "noise", " "], (400, 100), None)
    assert ocr.words == ["GrandTotal", "40,150", "noise"]                  # the empty reading is left out
    t = transfer(ocr, truth)
    assert t.tags == ["B-total", "I-total", "O"]
    assert counts(truth.fields(), decode(t.words, t.tags)) == (1, 0, 0)    # spaces do not matter
