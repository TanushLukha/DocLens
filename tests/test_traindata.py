import numpy as np
import pytest

from doclens import datasets
from doclens.traindata import Batches, Sources, Spec, fits, number_text

pytestmark = pytest.mark.skipif(not __import__("PIL.features").features.check("raqm"),
                                reason="needs libraqm to draw Hindi")

LINES = [("deva", "भारत एक विशाल देश है"), ("deva", "इसकी राजधानी नई दिल्ली है"),
         ("latn", "The quick brown fox jumps"), ("mixed", "हाइड्रोजन (Hydrogen) एक तत्व है")]


def spec(tmp_path, **kw) -> Spec:
    datasets.write_tsv(tmp_path / "lines.tsv", ["script", "text"], [list(r) for r in LINES])
    text = "".join(t for _, t in LINES)
    chars = sorted(set(text) | set(text.upper()) | set("0123456789.,Rpx "))
    return Spec(alphabet="".join(chars), height=32, max_width=32 * 32, batch=4, pool_batches=2,
                weights={"lines": 1.0}, lines_file=str(tmp_path / "lines.tsv"), **kw)


def test_fits_counts_a_blank_between_equal_neighbours():
    assert fits(3, [1, 2, 3]) and not fits(3, [1, 1, 2]) and fits(4, [1, 1, 2]) and not fits(5, [])


def test_numbers_look_like_receipts():
    rng = np.random.default_rng(0)
    texts = [number_text(rng) for _ in range(200)]
    assert all(t and set(t) <= set("0123456789.,Rpx ") for t in texts)
    assert any(t.startswith("Rp ") for t in texts) and any(t.startswith("x") for t in texts)


def test_samples_are_drawn_the_same_way_for_the_same_seed(tmp_path):
    s = spec(tmp_path)
    src = Sources(s)
    a = src.sample(np.random.default_rng([0, 1]))
    b = src.sample(np.random.default_rng([0, 1]))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert a[0].shape[0] == 32 and a[0].dtype == np.float32


def test_batches_hold_images_of_similar_width_with_their_targets(tmp_path):
    stream = iter(Batches(spec(tmp_path)))
    for _ in range(4):
        b = next(stream)
        assert b["x"].shape[:3] == (4, 1, 32)
        assert int(b["widths"].max()) == b["x"].shape[3]
        assert int(b["target_lengths"].sum()) == len(b["targets"]) and int(b["targets"].min()) >= 1
        assert sum(b["sources"].values()) == 4
