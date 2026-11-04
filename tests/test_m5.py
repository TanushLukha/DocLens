import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "m5"))
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402

from doclens.fields import split_line  # noqa: E402
from doclens.pages import box_quad  # noqa: E402
from doclens.tesseract import parse_tsv  # noqa: E402

ROWS = [{"id": f"{s}/{i}", "set": s, "text": f"{s} {i}"} for s in ("a", "b") for i in range(5)]


def test_pick_keeps_file_order_and_a_limit_per_set():
    assert [r["id"] for r in common.pick(ROWS, limit=2)] == ["a/0", "a/1", "b/0", "b/1"]
    assert [r["id"] for r in common.pick(ROWS, ["b"], 1)] == ["b/0"]
    assert len(common.pick(ROWS)) == 10


def test_results_continue_a_run_and_drop_other_settings(tmp_path):
    res = common.Results("sys", "crops", "v1", root=tmp_path)
    common.run(res, ROWS[:3], lambda part: [{"text": r["text"].upper()} for r in part], batch=2)
    again = common.Results("sys", "crops", "v1", root=tmp_path)
    assert again.done == {"a/0", "a/1", "a/2"}
    seen = []
    common.run(again, ROWS[:4], lambda part: seen.extend(part) or [{"text": ""} for _ in part])
    assert [r["id"] for r in seen] == ["a/3"]                                   # only the image not yet read
    rows = [json.loads(line) for line in again.path.read_text().splitlines()]
    assert rows[0]["text"] == "A 0" and rows[0]["tag"] == "v1" and rows[0]["seconds"] >= 0
    other = common.Results("sys", "crops", "v2", root=tmp_path)                 # changed settings: read again
    assert other.done == set() and other.path.read_text() == ""


def test_a_failed_image_is_kept_with_no_text_and_many_in_a_row_stop_the_run(tmp_path):
    def read(part):
        if any(r["id"] == "a/1" for r in part):
            raise ValueError("cannot read")
        return [{"text": r["text"]} for r in part]

    res = common.Results("sys", "crops", "v1", root=tmp_path)
    common.run(res, ROWS[:4], read, batch=4)
    rows = {r["id"]: r for r in map(json.loads, res.path.read_text().splitlines())}
    assert rows["a/1"]["text"] == "" and "cannot read" in rows["a/1"]["error"]
    assert rows["a/2"]["text"] == "a 2" and "error" not in rows["a/2"]
    res = common.Results("sys", "pages", "v1", root=tmp_path)
    with pytest.raises(RuntimeError):
        common.run(res, ROWS * 2, lambda part: 1 / 0, batch=1)


def test_html_text_breaks_lines_at_blocks_and_spaces_cells():
    html = "<div data-label='Text'><p>Hello <b>world</b></p>Line&amp;2<br/>x</div><table><tr><td>a</td><td>1</td></tr></table>"
    assert common.html_text(html) == "Hello world\nLine&2\nx\na 1"


def test_memory_of_this_process():
    m = common.memory()
    assert m is not None and m["peak"] >= m["now"] > 0


def test_split_line_shares_the_box_by_length():
    words = split_line(box_quad(0, 0, 100, 10), "ab cd  efgh")                # two spaces become one
    assert [w for w, _ in words] == ["ab", "cd", "efgh"]
    assert np.allclose(words[1][1], box_quad(30, 0, 50, 10))
    one = box_quad(3.3, 1, 7.7, 9)
    assert split_line(one, "word")[0][1].tolist() == one.tolist()              # one word keeps its box exactly
    assert split_line(one, "  ") == []


def test_tesseract_words_with_their_own_boxes():
    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [head, "5\t1\t1\t1\t1\t1\t10\t5\t20\t10\t90\tHello", "5\t1\t1\t1\t1\t2\t40\t5\t30\t10\t90\tworld",
            "5\t1\t1\t1\t2\t1\t10\t30\t25\t10\t90\tnext"]
    lines, words = parse_tsv("\n".join(rows)), parse_tsv("\n".join(rows), words=True)
    assert lines.texts == ["Hello world", "next"] and words.texts == ["Hello", "world", "next"]
    assert words.quads[1].tolist() == box_quad(40, 5, 70, 15).tolist()


def test_rows_order_groups_lines_and_reads_them_left_to_right():
    pytest.importorskip("doctr")
    from doclens.doctr_lines import rows_order
    quads = [box_quad(300, 100, 380, 120), box_quad(10, 100, 90, 120), box_quad(10, 10, 90, 30),
             box_quad(120, 12, 200, 31)]
    assert rows_order(quads, (400, 200)) == [2, 3, 1, 0]
    assert rows_order([], (400, 200)) == []


def test_scores_use_only_the_images_every_system_read():
    import m5_score
    truth = [{"id": f"cord_word/{i}", "set": "cord_word", "text": t} for i, t in enumerate(["abc", "de", "f"])]
    outs = {"doclens": {"cord_word/0": {"set": "cord_word", "text": "abc"}, "cord_word/1": {"set": "cord_word", "text": "de"},
                        "cord_word/2": {"set": "cord_word", "text": "f"}},
            "easyocr": {"cord_word/0": {"set": "cord_word", "text": "abd"}, "cord_word/1": {"set": "cord_word", "text": "",
                                                                                         "error": "x"}},
            "surya": {}}
    res = m5_score.score_crops(truth, outs)["cord_word"]
    assert res["n"] == 2 and set(res["systems"]) == {"doclens", "easyocr"}
    e = res["systems"]["easyocr"]
    assert e["cer"] == pytest.approx(3 / 5) and e["errors"] == 1 and e["vs_doclens"]["diff"] == pytest.approx(3 / 5)


def test_pages_are_scored_over_designs_and_by_group():
    import m5_score
    truth = [{"id": f"synth/{i}", "set": "synth", "text": "one two", "meta": {"base": f"d{i // 2}", "columns": 1 + i % 2,
                                                                                "condition": "clean"}} for i in range(4)]
    outs = {"doclens": {r["id"]: {"set": "synth", "text": "one two"} for r in truth},
            "tesseract": {r["id"]: {"set": "synth", "text": "one" if r["meta"]["columns"] == 2 else "one two"} for r in truth}}
    res = m5_score.score_pages(truth, outs)["synth"]
    t = res["systems"]["tesseract"]
    assert res["designs"] == 2 and t["page_cer"]["rate"] == pytest.approx(8 / 28)
    assert t["groups"]["columns 1"] == 0 and t["groups"]["columns 2"] == pytest.approx(4 / 7)


def test_receipt_words_in_the_systems_order_and_in_rows():
    pytest.importorskip("doctr")
    import m5_score
    row = {"text": "TOTAL 9,100\nTAX 900", "lines": [{"quad": box_quad(10, 50, 110, 60).tolist(), "text": "TOTAL 9,100"},
                                                      {"quad": box_quad(10, 10, 70, 20).tolist(), "text": "TAX 900"}]}
    own = m5_score.own_order("7", row, (200, 100))
    assert own.words == ["TOTAL", "9,100", "TAX", "900"]
    rows = m5_score.rows_receipt("7", row, (200, 100))
    assert rows.words == ["TAX", "900", "TOTAL", "9,100"]
    assert m5_score.rows_receipt("7", {"text": "x", "blocks": []}, (200, 100)) is None   # no boxes for lines


def test_the_reference_check_pairs_the_two_on_the_crops_both_read():
    import m5_score
    truth = [{"id": f"synth/{i}", "set": "synth", "text": t} for i, t in enumerate(["abcd", "ef", "gh"])]
    served = {"synth/0": {"text": "abcx"}, "synth/1": {"text": "ef"}, "synth/2": {"text": ""}}
    ref = {"synth/0": {"text": "abcd", "seconds": 2.0}, "synth/1": {"text": "ex", "seconds": 4.0}}
    res = m5_score.score_reference(truth, served, ref)["synth"]
    assert res["n"] == 2 and res["reference_seconds"] == 3.0                    # synth/2 was not read by both
    assert res["served"]["cer"] == pytest.approx(1 / 6) and res["reference"]["cer"] == pytest.approx(1 / 6)
    assert res["served_vs_reference"]["diff"] == pytest.approx(0.0)
    assert m5_score.score_reference(truth, served, {}) == {}


def test_the_probe_sizes_images_as_the_model_does():
    import probe
    assert probe.smart_resize(72, 90) == (308, 392) and probe.smart_resize(3508, 2480) == (1176, 840)
    h, w = probe.smart_resize(30, 268)
    assert h % 28 == 0 and w % 28 == 0 and h * w >= probe.MIN_PIXELS


def test_runs_of_a_task_add_up_and_keep_the_higher_peak():
    first = {"tag": "t", "read": 10, "load_seconds": 5.0, "memory": {"now": 1, "peak": 8}, "servers": {"1": {"peak": 2}}}
    second = {"tag": "t", "read": 5, "load_seconds": 7.0, "memory": {"now": 1, "peak": 9}, "servers": {"2": {"peak": 3}}}
    both = common.merged(first, second)
    assert both["read"] == 15 and both["load_seconds"] == 5.0 and both["runs"] == 2
    assert both["memory"]["peak"] == 9 and both["servers"] == {"2": {"peak": 3}}     # one run's process with its own server
    assert common.merged(first, {**second, "read": 0}) is first                    # read nothing: the record stays
    assert common.merged(first, {**second, "tag": "u"})["read"] == 5               # other settings: a new record


def test_blocks_written_as_html_become_text():
    import run_paddleocr_vl
    table = "<table><tr><td>901016</td><td>-TICKET CP</td></tr><tr><td>2</td><td>60.000</td></tr></table>"
    assert run_paddleocr_vl.plain("table", table) == "901016 -TICKET CP\n2 60.000"
    assert run_paddleocr_vl.plain("text", "TOTAL < 9,100 & more") == "TOTAL < 9,100 & more"


def test_a_diagnosis_reads_columns_one_after_the_other_and_counts_invented_text():
    pytest.importorskip("doctr")
    import m5_diagnose
    q = lambda x0, y0, x1, y1: box_quad(x0, y0, x1, y1).tolist()                # noqa: E731
    items = [{"text": "right one", "quad": q(60, 20, 95, 25)}, {"text": "left", "quad": q(5, 20, 20, 25)},
             {"text": "one", "quad": q(22, 20, 40, 25)}, {"text": "Title", "quad": q(5, 5, 95, 10)},
             {"text": "left two", "quad": q(5, 30, 40, 35)}, {"text": "right two", "quad": q(60, 30, 95, 35)}]
    assert m5_diagnose.by_columns(items, (100, 100)).split("\n") == ["Title", "left", "one", "left two", "right one",
                                                                      "right two"]
    truth = [{"id": "cord_word/0", "set": "cord_word", "text": "PB1"}, {"id": "cord_word/1", "set": "cord_word", "text": "2"}]
    outs = {"surya": {"cord_word/0": {"text": "DEPARTMENT OF JUSTICE"}, "cord_word/1": {"text": "2"}}}
    res = m5_diagnose.invented(truth, [], outs, {})["crops"]["cord_word"]["surya"]
    assert res["too_long"] == 1 and res["share_of_edits"] == 1.0
