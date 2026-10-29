"""The recognizer: a CRNN (a convolutional network, then a bidirectional LSTM) trained with CTC.

    line image, 1 x H x W  →  CNN  →  W/4 feature columns  →  BiLSTM  →  a score per class per column

The CNN reads small neighbourhoods of pixels and turns every 4 pixel wide slice of the line into
one feature vector; the LSTM reads those vectors left to right and right to left, so each column
knows its context (a Devanagari vowel sign drawn before its consonant is written after it); the
last layer scores every class of the alphabet for every column. CTC loss trains this without
knowing where each character sits: it sums over every way of spreading the true text over the
columns, with blanks between and repeats collapsed.

Images are scaled to a fixed height, keeping their aspect ratio, so the number of columns grows
with the length of the line. `prepare` is the one place that turns a crop into network input,
used by training, evaluation and later the service, so they can never disagree.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .alphabet import BLANK, Alphabet
from .preprocess import gray

DOWNSAMPLE = 4          # columns: one per 4 pixels of width


@dataclass
class CRNNConfig:
    n_classes: int
    height: int = 48
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    max_aspect: int = 32            # wider images are squeezed to height * max_aspect pixels

    @property
    def max_width(self) -> int:
        return self.height * self.max_aspect


def _conv(cin: int, cout: int, kernel=3, padding=1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, kernel, padding=padding, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, cfg: CRNNConfig):
        super().__init__()
        if cfg.height % 16:
            raise ValueError("height must be a multiple of 16 (four halvings, then one row)")
        self.cfg = cfg
        c0, c1, c2, c3 = cfg.channels
        self.cnn = nn.Sequential(
            *_conv(1, c0), nn.MaxPool2d(2, 2),                       # H/2, W/2
            *_conv(c0, c1), nn.MaxPool2d(2, 2),                      # H/4, W/4
            *_conv(c1, c2), *_conv(c2, c2), nn.MaxPool2d((2, 1)),    # H/8: from here on only height shrinks
            *_conv(c2, c3), *_conv(c3, c3), nn.MaxPool2d((2, 1)),    # H/16
            *_conv(c3, c3, kernel=(cfg.height // 16, 1), padding=0),  # one row: a feature vector per column
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.rnn = nn.LSTM(c3, cfg.hidden, num_layers=cfg.layers, bidirectional=True,
                           dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Linear(2 * cfg.hidden, cfg.n_classes)

    def forward(self, x: torch.Tensor, widths: torch.Tensor, packed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """x: B x 1 x H x W (padded on the right), widths: true widths in pixels.
        Returns (logits T x B x C, columns per image)."""
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)              # T x B x channels
        lengths = torch.clamp(torch.div(widths, DOWNSAMPLE, rounding_mode="floor"), 1, f.size(0))
        f = self.drop(f)
        if packed:
            # The LSTM then starts and stops at each image's own end, so the right to left pass does
            # not cross the padding of a wider neighbour first. Only the convolutions' borders still
            # see that padding, which moves the scores by a hair (tests/test_crnn.py measures it).
            seq = nn.utils.rnn.pack_padded_sequence(f, lengths.cpu(), enforce_sorted=False)
            out, _ = self.rnn(seq)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=f.size(0))
        else:
            out, _ = self.rnn(f)
        return self.head(self.drop(out)), lengths


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# input
def prepare(img: np.ndarray, height: int, max_width: int) -> np.ndarray:
    """A crop as network input: grey, `height` pixels tall, contrast stretched, float32 from minus 1 to 1.

    The contrast stretch maps the 2nd percentile (ink) to minus 1 and the 98th (paper) to plus 1, so faded
    thermal print and a dark shadowed scan reach the network on the same scale. When the two are
    close (a blank crop) the range is kept at 32 grey levels at least, so noise is not blown up.
    """
    img = gray(img)
    h, w = img.shape
    new_w = int(np.clip(round(w * height / max(h, 1)), DOWNSAMPLE * 2, max_width))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, height), interpolation=interp).astype(np.float32)
    lo, hi = np.percentile(img, (2, 98))
    if hi - lo < 32:
        mid = (hi + lo) / 2
        lo, hi = mid - 16, mid + 16
    x = np.clip((img - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
    return ((x - 0.5) / 0.5).astype(np.float32)


def collate(images: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepared images into one padded batch (padding is paper: plus 1) and their widths."""
    h = images[0].shape[0]
    w = max(im.shape[1] for im in images)
    x = np.ones((len(images), 1, h, w), np.float32)
    for i, im in enumerate(images):
        x[i, 0, :, : im.shape[1]] = im
    return torch.from_numpy(x), torch.tensor([im.shape[1] for im in images], dtype=torch.long)


def ctc_loss(logits: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor,
             on_cpu: bool = True) -> torch.Tensor:
    """Mean CTC loss per image, each divided by its text length (PyTorch's "mean").

    log_softmax runs where the model runs; the CTC sum itself runs on the CPU by default: it is a
    small dynamic programme, cheap next to the network, and the CPU kernel exists on every
    PyTorch build (scripts/bench_m2.py checks the Apple GPU kernel and times both).
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    if on_cpu:
        logp, lengths, targets, target_lengths = logp.cpu(), lengths.cpu(), targets.cpu(), target_lengths.cpu()
    return F.ctc_loss(logp, targets, lengths, target_lengths, blank=BLANK, reduction="mean", zero_infinity=True)


# devices and checkpoints
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save(path: str | Path, model: CRNN, alphabet: Alphabet, **extra) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": dataclasses.asdict(model.cfg), "model": state, "alphabet": "".join(alphabet.chars),
                **extra}, path)


def load(path: str | Path, device: torch.device | str = "cpu") -> tuple[CRNN, Alphabet, dict]:
    """weights_only: a checkpoint may hold tensors and plain values only, so loading one runs no code."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    cfg["channels"] = tuple(cfg["channels"])
    model = CRNN(CRNNConfig(**cfg))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), Alphabet(ckpt["alphabet"]), ckpt
