import numpy as np
import pytest
import torch

from doclens import crnn, fonts
from doclens.alphabet import Alphabet
from doclens.crnn import CRNN, CRNNConfig, collate, prepare
from doclens.ctc import greedy

SMALL = {"channels": (8, 16, 32, 32), "hidden": 32}


def test_prepare_scales_to_height_and_stretches_contrast():
    img = np.full((60, 300), 200, np.uint8)
    img[20:40, 30:270] = 90
    x = prepare(img, 48, 48 * 32)
    assert x.shape == (48, 240) and x.dtype == np.float32
    assert x.min() == pytest.approx(-1.0) and x.max() == pytest.approx(1.0)
    blank = prepare(np.full((40, 10), 250, np.uint8), 32, 1024)       # nothing to stretch, and very narrow
    assert blank.shape == (32, 8) and np.isfinite(blank).all()
    assert prepare(np.zeros((10, 5000), np.uint8), 32, 1024).shape == (32, 1024)


def test_columns_follow_the_width_and_packing_keeps_padding_out():
    torch.manual_seed(0)
    model = CRNN(CRNNConfig(n_classes=10, height=32, **SMALL)).eval()
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-1, 1, (32, w)).astype(np.float32) for w in (64, 200))
    x, widths = collate([a, b])
    with torch.no_grad():
        logits, lengths = model(x, widths)
        plain, _ = model(x, widths, packed=False)
        alone, _ = model(*collate([a]))
    assert logits.shape == (50, 2, 10) and lengths.tolist() == [16, 50]
    # Read next to a wider image, `a` is padded with paper. Without packing, the right to left LSTM
    # crosses 34 columns of padding before reaching it; packed, it starts at a's own last column, and
    # only the convolutions' borders still see the padding (a few pixels at the right edge).
    packed_gap = (logits[:16, 0] - alone[:, 0]).abs().max()
    plain_gap = (plain[:16, 0] - alone[:, 0]).abs().max()
    assert packed_gap < 1e-3 and packed_gap < plain_gap / 10


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = CRNN(CRNNConfig(n_classes=4, height=32, **SMALL)).eval()
    abc = Alphabet(["a", "b", "c"])
    crnn.save(tmp_path / "m.pt", model, abc, step=7)
    loaded, abc2, ckpt = crnn.load(tmp_path / "m.pt")
    x, w = collate([np.zeros((32, 40), np.float32)])
    with torch.no_grad():
        assert torch.equal(model(x, w)[0], loaded(x, w)[0])
    assert abc2.chars == abc.chars and ckpt["step"] == 7


@pytest.mark.skipif(not __import__("PIL.features").features.check("raqm"), reason="needs libraqm to draw Hindi")
def test_a_small_crnn_learns_to_read_a_few_words():
    """The whole training path (drawing, preparing, CTC, decoding) on four words: it must memorise them."""
    from doclens.render import render_line
    torch.manual_seed(0)
    words = ["भारत", "क्षत्रिय", "TOTAL 450", "नई दिल्ली"]
    abc = Alphabet(sorted(set("".join(words))))
    deva = next(f for f in fonts.load("train", "deva") if f.name == "Noto Sans Devanagari")
    latn = next(f for f in fonts.load("train", "latn") if f.name == "Liberation Sans")
    x, widths = collate([prepare(render_line(w, deva, latn, 28).image, 32, 1024) for w in words])
    targets = torch.tensor([c for w in words for c in abc.encode(w)])
    lengths = torch.tensor([len(w) for w in words])
    model = CRNN(CRNNConfig(n_classes=len(abc), height=32, dropout=0.0, channels=(16, 32, 64, 64), hidden=64))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(250):
        logits, cols = model(x, widths)
        loss = crnn.ctc_loss(logits, cols, targets, lengths)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, cols = model(x, widths)
    logp = torch.log_softmax(logits, -1).numpy()
    read = [abc.decode(greedy(logp[: cols[i], i])) for i in range(len(words))]
    assert read == words
