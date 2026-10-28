import numpy as np
import pytest

from doclens import degrade, fonts, preprocess
from doclens.render import render_line


@pytest.fixture(scope="module")
def line():
    f = next(f for f in fonts.load(script="latn") if f.name == "Liberation Serif")
    return render_line("The quick brown fox jumps over the lazy dog, 4521.", f, f, 32).image


@pytest.mark.parametrize("angle", [3.0, -4.0, 1.5])
def test_deskew_finds_and_removes_the_rotation(line, angle):
    tilted = degrade.rotate(line, angle, fill=245)
    assert abs(preprocess.skew_angle(tilted) + angle) < 0.3
    assert abs(preprocess.skew_angle(preprocess.deskew(tilted))) < 0.3


def test_sauvola_is_binary_and_keeps_text_under_a_shadow(line):
    shaded = degrade.shadow(line, np.random.default_rng(0), (0.6, 0.6))
    out = preprocess.sauvola(shaded)
    assert set(np.unique(out)) <= {0, 255}
    # the shaded half must keep about as much ink as the lit half
    h, w = out.shape
    left, right = (out[:, : w // 2] == 0).mean(), (out[:, w // 2:] == 0).mean()
    assert min(left, right) / max(left, right) > 0.5


def test_every_pipeline_returns_grayscale_uint8(line):
    for name in preprocess.PIPELINES:
        out = preprocess.apply(name, line)
        assert out.dtype == np.uint8 and out.ndim == 2


def test_conditions_are_seeded(line):
    for name, fn in degrade.CONDITIONS.items():
        a = fn(line, np.random.default_rng(7))
        b = fn(line, np.random.default_rng(7))
        assert np.array_equal(a, b), name
