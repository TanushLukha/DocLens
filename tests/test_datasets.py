import json

import cv2
import numpy as np
import pytest

from doclens import datasets
from doclens.corpus import collect_lines, to_lines


def _png(path, h=40, w=120):
    img = np.full((h, w), 230, np.uint8)
    cv2.putText(img, "ab", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


@pytest.mark.parametrize("layout", ["images_prefix", "bare_names"])
def test_mozhi_reader_finds_images_in_either_layout(tmp_path, layout):
    split = tmp_path / "test"
    names = ["w1.jpeg", "w2.jpeg"]           # Mozhi ships JPEG word crops in images/
    for n in names:
        _png(split / "images" / n)
    rel = [f"images/{n}" if layout == "images_prefix" else n for n in names]
    (split / "test_gt.txt").write_text(f"{rel[0]}\tभारत\n{rel[1]}\tदेश\n", encoding="utf8")
    items = datasets.mozhi("test", root=tmp_path)
    assert [it.text for it in items] == ["भारत", "देश"]
    assert all(it.image().shape == (60, 140) for it in items)          # a margin of a quarter of the height
    bare = datasets.mozhi("test", root=tmp_path, margin=0.0)
    assert all(it.image().shape == (40, 120) for it in bare)


def test_margin_is_paper_coloured():
    img = np.full((40, 100), 230, np.uint8)
    img[10:30, 0:100] = 20                          # ink right up to both side edges
    out = datasets.add_margin(img, 0.25)
    assert out.shape == (60, 120)
    assert (out[:, :10] == 230).all() and (out[:10, :] == 230).all()


def test_mozhi_reader_accepts_space_separated_lines_and_refuses_an_empty_file(tmp_path):
    split = tmp_path / "val"
    _png(split / "images" / "1.jpeg")
    (split / "val_gt.txt").write_text("1.jpeg भारत\n", encoding="utf8")
    assert [it.text for it in datasets.mozhi("val", root=tmp_path)] == ["भारत"]
    (split / "val_gt.txt").write_text("\n", encoding="utf8")
    with pytest.raises(ValueError):
        datasets.mozhi("val", root=tmp_path)


def test_cord_words_and_lines(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    img = np.full((200, 300), 240, np.uint8)
    ok, png = cv2.imencode(".png", img)

    def quad(x1, y1, x2, y2):
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y1, "x3": x2, "y3": y2, "x4": x1, "y4": y2}
    gt = {"valid_line": [
        {"category": "menu.nm", "words": [{"quad": quad(120, 10, 200, 40), "text": "TEA", "row_id": 1},
                                          {"quad": quad(10, 10, 100, 40), "text": "GREEN", "row_id": 1}]},
        {"category": "total.total_price", "words": [{"quad": quad(10, 60, 90, 90), "text": "12,000", "row_id": 2}]},
    ]}
    table = pa.table({"image": [{"bytes": png.tobytes(), "path": "r0.png"}], "ground_truth": [json.dumps(gt)]})
    (tmp_path / "data").mkdir()
    pq.write_table(table, tmp_path / "data" / "test-00000-of-00001.parquet")
    words = datasets.cord("test", "word", root=tmp_path)
    lines = datasets.cord("test", "line", root=tmp_path)
    assert sorted(w.text for w in words) == ["12,000", "GREEN", "TEA"]
    assert [l.text for l in lines] == ["GREEN TEA", "12,000"]       # words of a row, left to right
    assert all(it.image().ndim == 2 and it.image().size > 0 for it in words + lines)


def test_synth_labels_round_trip(tmp_path):
    _png(tmp_path / "s" / "images" / "a.png")
    rows = [{"file": "images/a.png", "text": "क्षत्रिय \"quoted\" text", "condition": "clean", "fonts": "Noto",
             "script": "mixed", "size": 30}]
    datasets.write_synth(tmp_path / "s", rows)
    items = datasets.synth("s", root=tmp_path)
    assert items[0].text == rows[0]["text"] and items[0].meta["size"] == 30


def test_lines_are_packed_within_bounds():
    import random
    rng = random.Random(0)
    s = " ".join(["शब्द"] * 200)
    lines = list(to_lines(s, rng, 25, 70))
    assert " ".join(lines) == s
    assert all(len(l) <= 70 for l in lines[:-1])


def test_collect_lines_groups_by_script(tmp_path):
    docs = [{"text": "भारत एक विशाल देश है। इसकी राजधानी नई दिल्ली है।", "src": "wiki_hi"},
            {"text": "हाइड्रोजन (अंग्रेज़ी: Hydrogen) एक रासायनिक तत्व है।", "src": "wiki_hi"},
            {"text": "kal ka weather kaisa hoga yaar batao", "src": "hinglish_top"}]
    (tmp_path / "val.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf8")
    out = collect_lines(tmp_path, "val", None, seed=0)
    assert out["deva"] and out["latn"] and out["mixed"]
    assert any("Hydrogen" in l for l in out["mixed"])
