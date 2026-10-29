"""Reading crops with a trained CRNN: batching, the network, then greedy or beam decoding.

Images are sorted by width before batching, so each batch holds crops of similar width, and
results come back in the order the images were given.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import crnn
from .ctc import beam_search, greedy


class Recognizer:
    def __init__(self, model: crnn.CRNN, alphabet, device: torch.device | str = "cpu"):
        self.model, self.alphabet, self.device = model.to(device).eval(), alphabet, torch.device(device)
        self.height, self.max_width = model.cfg.height, model.cfg.max_width

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Recognizer:
        dev = crnn.pick_device(device)
        model, alphabet, _ = crnn.load(path, dev)
        return cls(model, alphabet, dev)

    @torch.no_grad()
    def logprobs(self, images: Sequence[np.ndarray], batch_size: int = 64) -> Iterator[tuple[int, np.ndarray]]:
        """(index, T x C log probabilities) for every image, in batches of similar width."""
        order = sorted(range(len(images)), key=lambda i: images[i].shape[1] / max(images[i].shape[0], 1))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            x, widths = crnn.collate([crnn.prepare(images[i], self.height, self.max_width) for i in idx])
            logits, lengths = self.model(x.to(self.device), widths.to(self.device))
            logp = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for j, i in enumerate(idx):
                yield i, logp[: int(lengths[j]), j]

    def _each(self, images, decode: Callable[[np.ndarray], object], batch_size: int) -> list:
        out: list = [None] * len(images)
        for i, lp in self.logprobs(images, batch_size):
            out[i] = decode(lp)
        return out

    def read(self, images: Sequence[np.ndarray], batch_size: int = 64) -> list[str]:
        """Greedy decoding."""
        return self._each(images, lambda lp: self.alphabet.decode(greedy(lp)), batch_size)

    def nbest(self, images: Sequence[np.ndarray], beam: int = 10, batch_size: int = 64) -> list[list[tuple[str, float]]]:
        """Prefix beam search: up to `beam` (text, log probability) candidates per image, best first."""
        return self._each(images, lambda lp: [(self.alphabet.decode(k), s) for k, s in beam_search(lp, beam)],
                          batch_size)
