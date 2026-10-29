import numpy as np
import pytest
import torch

from doclens import crnn, fonts
from doclens.alphabet import Alphabet
from doclens.crnn import CRNN, BiLSTM, CRNNConfig, collate, prepare
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


def test_bilstm_is_a_bidirectional_lstm_and_padding_never_reaches_a_real_column():
    torch.manual_seed(0)
    ours = BiLSTM(8, 16, 2, dropout=0.0).eval()
    ref = torch.nn.LSTM(8, 16, num_layers=2, bidirectional=True).eval()
    with torch.no_grad():                       # the same weights, in PyTorch's own layout
        for i in range(2):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(ref, f"{name}_l{i}").copy_(getattr(ours.fwd[i], f"{name}_l0"))
                getattr(ref, f"{name}_l{i}_reverse").copy_(getattr(ours.bwd[i], f"{name}_l0"))
    x = torch.randn(12, 3, 8)
    with torch.no_grad():
        full = ours(x, torch.tensor([12, 12, 12]))
        assert torch.allclose(full, ref(x)[0], atol=1e-5)          # no padding: exactly PyTorch's LSTM
        short = ours(x, torch.tensor([12, 7, 12]))                   # sequence 1 ends after 7 columns
        alone = ours(x[:7, 1:2], torch.tensor([7]))
    assert torch.allclose(short[:7, 1], alone[:, 0], atol=1e-6)      # its 5 padded columns change nothing


def test_columns_follow_the_width_and_a_wider_neighbour_barely_matters():
    torch.manual_seed(0)
    model = CRNN(CRNNConfig(n_classes=10, height=32, **SMALL)).eval()
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-1, 1, (32, w)).astype(np.float32) for w in (64, 200))
    x, widths = collate([a, b])
    with torch.no_grad():
        logits, lengths = model(x, widths)
        alone, _ = model(*collate([a]))
    assert x.shape[3] == 256 and logits.shape == (64, 2, 10) and lengths.tolist() == [16, 50]
    # Next to a wider image, a is padded with paper. The LSTM never reads that padding; only the
    # convolutions at a's right edge see it, through a few pixels.
    assert (logits[:16, 0] - alone[:, 0]).abs().max() < 1e-3


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
