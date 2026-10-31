import numpy as np
import pytest

pytest.importorskip("doctr")

from doclens.doctr_lines import DoctrLines  # noqa: E402


def test_words_become_lines_in_both_orders():
    """docTR's grouping and ordering, fed words as its detector reports them (relative corners and a
    score row), without loading the detector."""
    from doctr.models.builder import DocumentBuilder

    def word(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [0.9, 0.9]], np.float32) / [1000, 1000]

    words = [word(100 + 110 * k, 150 + 60 * i, 190 + 110 * k, 185 + 60 * i) for i in range(3) for k in range(3)]
    words += [word(550 + 110 * k, 150 + 60 * i, 640 + 110 * k, 185 + 60 * i) for i in range(3) for k in range(3)]
    for w in words:
        w[4] = 0.9                                            # the score row, as docTR returns it
    lines = DoctrLines.__new__(DoctrLines)
    lines.det = lambda pages: [{"words": np.stack(words)}]
    lines.builders = {"order": DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=True),
                      "rows": DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=False)}
    out = lines(np.full((1000, 1000), 255, np.uint8))
    starts = {k: [(round(float(q[0][0])), round(float(q[0][1]))) for q in v] for k, v in out.items()}
    assert starts["order"] == [(100, 150), (100, 210), (100, 270), (550, 150), (550, 210), (550, 270)]   # columns
    assert starts["rows"] == [(100, 150), (550, 150), (100, 210), (550, 210), (100, 270), (550, 270)]    # rows


def test_words_come_in_rows_left_to_right():
    from doctr.models.builder import DocumentBuilder

    def word(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [0.9, 0.9]], np.float32) / [1000, 1000]

    words = [word(100 + 110 * k, 150 + 60 * i, 190 + 110 * k, 185 + 60 * i) for i in range(2) for k in range(3)]
    for w in words:
        w[4] = 0.9
    lines = DoctrLines.__new__(DoctrLines)
    lines.det = lambda pages: [{"words": np.stack(words)}]
    lines.builders = {"rows": DocumentBuilder(resolve_lines=True, resolve_blocks=False, keep_reading_order=False)}
    starts = [(round(float(q[0][0])), round(float(q[0][1]))) for q in lines.words(np.full((1000, 1000), 255, np.uint8))]
    assert starts == [(100, 150), (210, 150), (320, 150), (100, 210), (210, 210), (320, 210)]
