"""Reading crops with a trained CRNN: batching, the network, then greedy or beam decoding.

Each image goes to its width class (crnn.width_classes) and is read in a batch of that width and a
fixed size, the last batch of a class filled with blank images, so the GPU meets five shapes only.
Results come back in the order the images were given.
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
        """(index, T x C log probabilities) for every image."""
        classes = crnn.width_classes(self.height)
        groups: dict[int, list[tuple[int, np.ndarray]]] = {c: [] for c in classes}
        for i, im in enumerate(images):
            x = crnn.prepare(im, self.height, self.max_width)
            c = crnn.width_class(x.shape[1], classes)
            groups[c].append((i, x))
            if len(groups[c]) == batch_size:
                yield from self._run(groups[c], c, batch_size)
                groups[c] = []
        for c, rest in groups.items():
            if rest:
                yield from self._run(rest, c, batch_size)

    def _run(self, batch: list[tuple[int, np.ndarray]], width: int, batch_size: int):
        blank = np.ones((self.height, crnn.DOWNSAMPLE * 2), np.float32)
        images = [x for _, x in batch] + [blank] * (batch_size - len(batch))
        x, widths = crnn.collate(images, width)
        logits, lengths = self.model(x.to(self.device), widths.to(self.device))
        logp = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
        for j, (i, _) in enumerate(batch):
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
