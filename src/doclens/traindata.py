"""Training batches for the recognizer, made fresh at every step from three sources.

* lines: text from TinyGPT's training documents and English sentences (data/m2/train_lines.tsv,
  made by scripts/prepare_m2.py), drawn with the 29 training fonts at random sizes and colours,
  then left clean or damaged by one or two of degrade.CONDITIONS. Sometimes only a few words of a
  line are drawn, so the network also meets word sized crops; sometimes a Latin line is put in
  capitals, or replaced by a price or quantity, as on receipts.
* mozhi: the Mozhi training words, with the same border of paper the test words get.
* cord: word and line crops of CORD's training receipts (data/m2/cord_train/).

The test fonts, Mozhi's validation and test words and CORD's validation and test receipts are
never read here.

Each DataLoader worker builds whole batches: it draws a pool of samples, sorts them by width
and cuts the pool into batches, so a batch holds images of similar width and little compute is
spent on padding. A pool's random generator is seeded by (seed, pool number), so the stream is
the same on every run and a resumed run continues with pools it has not seen.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import DATA, datasets, fonts
from .alphabet import Alphabet
from .crnn import DOWNSAMPLE, collate, prepare
from .degrade import CONDITIONS
from .render import render_line

M2 = DATA / "m2"
DAMAGE = [c for c in CONDITIONS if c != "clean"]


@dataclass
class Spec:
    """Everything a worker needs, as plain values (workers are separate processes)."""
    alphabet: str
    height: int
    max_width: int
    batch: int = 64
    pool_batches: int = 8
    weights: dict = field(default_factory=lambda: {"lines": 0.5, "mozhi": 0.3, "cord": 0.2})
    script_weights: dict = field(default_factory=lambda: {"deva": 0.5, "latn": 0.3, "mixed": 0.2})
    seed: int = 0
    lines_file: str = str(M2 / "train_lines.tsv")
    mozhi_root: str = str(DATA / "mozhi" / "hindi")
    cord_dir: str = str(M2 / "cord_train")
    p_clean: float = 0.3          # synthetic lines left undamaged
    p_second: float = 0.2         # a second damage on top of the first
    p_real_damage: float = 0.3    # real crops damaged once
    p_span: float = 0.35          # draw only 1 to 4 words of a line
    p_caps: float = 0.15          # a Latin line in capitals
    p_number: float = 0.05        # a price or quantity instead of a line


def read_lines(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"deva": [], "latn": [], "mixed": []}
    for r in datasets.read_tsv(Path(path)):
        out[r["script"]].append(r["text"])
    return out


def number_text(rng: np.random.Generator) -> str:
    """A price, quantity or code in the styles receipts use: 12,000  45.500  Rp 120.000  x2  3 x 15,000."""
    n = int(rng.choice([rng.integers(1, 100), rng.integers(100, 100_000), rng.integers(1_000, 10_000_000)]))
    sep = "," if rng.random() < 0.5 else "."
    s = f"{n:,}".replace(",", sep)
    r = rng.random()
    if r < 0.2:
        return "Rp " + s
    if r < 0.3:
        return f"x{int(rng.integers(1, 20))}"
    if r < 0.4:
        return f"{int(rng.integers(1, 10))} x {s}"
    if r < 0.5:
        return s + (".00" if sep == "," else ",00")
    return s


class Sources:
    """The three sources, loaded once per worker."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self.alphabet = Alphabet(spec.alphabet)
        w = {k: v for k, v in spec.weights.items() if v > 0}
        self.lines = read_lines(spec.lines_file) if "lines" in w else None
        self.deva_fonts = fonts.load("train", "deva")
        self.latn_fonts = fonts.load("train", "latn")
        self.mozhi = datasets.mozhi("train", root=Path(spec.mozhi_root)) if "mozhi" in w else []
        self.cord = datasets.crops(Path(spec.cord_dir)) if "cord" in w else []
        if "mozhi" in w and not self.mozhi:
            raise FileNotFoundError("no Mozhi training words: python scripts/get_data.py --only mozhi_train")
        if "cord" in w and not self.cord:
            raise FileNotFoundError("no CORD training crops: python scripts/prepare_m2.py")
        self.names = list(w)
        p = np.array([w[k] for k in self.names], dtype=np.float64)
        self.p = p / p.sum()
        self.skipped = 0

    # one sample from each source: (uint8 image, text) or None
    def line(self, rng: np.random.Generator):
        s = self.spec
        if rng.random() < s.p_number:
            text, script = number_text(rng), "latn"
        else:
            names = list(s.script_weights)
            script = names[int(rng.choice(len(names), p=np.array([s.script_weights[k] for k in names])))]
            pool = self.lines[script] if self.lines else []
            if not pool:
                return None
            text = pool[int(rng.integers(len(pool)))]
            words = text.split()
            if rng.random() < s.p_span and len(words) > 1:
                k = int(rng.integers(1, min(4, len(words)) + 1))
                i = int(rng.integers(0, len(words) - k + 1))
                text = " ".join(words[i:i + k])
            if script == "latn" and rng.random() < s.p_caps:
                text = text.upper()
        if not text or not self.alphabet.covers(text):
            return None
        deva = self.deva_fonts[int(rng.integers(len(self.deva_fonts)))]
        latn = self.latn_fonts[int(rng.integers(len(self.latn_fonts)))]
        pad = (int(rng.integers(2, 21)), int(rng.integers(2, 17)), int(rng.integers(2, 21)), int(rng.integers(2, 17)))
        try:
            r = render_line(text, deva, latn, int(rng.integers(20, 45)), latin_in_deva_font=bool(rng.random() < 0.5),
                            ink=int(rng.integers(0, 81)), paper=int(rng.integers(180, 256)), pad=pad)
        except ValueError:                 # a character none of the two fonts has
            return None
        img = r.image
        if rng.random() >= s.p_clean:
            img = CONDITIONS[DAMAGE[int(rng.integers(len(DAMAGE)))]](img, rng)
            if rng.random() < s.p_second:
                img = CONDITIONS[DAMAGE[int(rng.integers(len(DAMAGE)))]](img, rng)
        return img, text

    def real(self, items: list, rng: np.random.Generator):
        it = items[int(rng.integers(len(items)))]
        if not it.text or not self.alphabet.covers(it.text):
            return None
        img = it.image()
        if rng.random() < self.spec.p_real_damage:
            img = CONDITIONS[DAMAGE[int(rng.integers(len(DAMAGE)))]](img, rng)
        return img, it.text

    def sample(self, rng: np.random.Generator):
        """(prepared image, class indices, source name); retries until a sample fits."""
        for _ in range(50):
            name = self.names[int(rng.choice(len(self.names), p=self.p))]
            got = self.line(rng) if name == "lines" else self.real(self.mozhi if name == "mozhi" else self.cord, rng)
            if got is None:
                self.skipped += 1
                continue
            img, text = got
            x = prepare(img, self.spec.height, self.spec.max_width)
            target = self.alphabet.encode(text)
            if not fits(x.shape[1] // DOWNSAMPLE, target):
                self.skipped += 1
                continue
            return x, target, name
        raise RuntimeError("50 samples in a row did not fit: check the alphabet and the data")

    def pool(self, rng: np.random.Generator):
        """Batches of similar width from one pool of samples, in random order."""
        n = self.spec.batch * self.spec.pool_batches
        samples = sorted((self.sample(rng) for _ in range(n)), key=lambda s: s[0].shape[1])
        batches = [samples[i:i + self.spec.batch] for i in range(0, n, self.spec.batch)]
        order = rng.permutation(len(batches))
        for k in order:
            yield make_batch(batches[int(k)])


def fits(columns: int, target: list[int]) -> bool:
    """CTC needs a column per character, plus one for a blank between two equal neighbours."""
    repeats = sum(1 for a, b in zip(target, target[1:]) if a == b)
    return 0 < len(target) and len(target) + repeats <= columns


def make_batch(samples) -> dict:
    x, widths = collate([s[0] for s in samples])
    targets = torch.tensor([c for s in samples for c in s[1]], dtype=torch.long)
    lengths = torch.tensor([len(s[1]) for s in samples], dtype=torch.long)
    sources: dict[str, int] = {}
    for s in samples:
        sources[s[2]] = sources.get(s[2], 0) + 1
    return {"x": x, "widths": widths, "targets": targets, "target_lengths": lengths, "sources": sources}


class Batches(IterableDataset):
    """An endless stream of batches; pool p is made by worker p mod workers."""

    def __init__(self, spec: Spec, start_pool: int = 0):
        self.spec = spec
        self.start_pool = start_pool

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        cv2.setNumThreads(1)                # one OpenCV thread per worker: the workers are the parallelism
        src = Sources(self.spec)
        p = self.start_pool + wid
        while True:
            yield from src.pool(np.random.default_rng([self.spec.seed, p]))
            p += nw


# validation sets: model selection only, never the test sets
def val_sets(mozhi_n: int = 2000) -> dict[str, list]:
    """Mozhi validation (a fixed sample of 2,000 words), synthetic validation lines (training fonts,
    held out text that is not in the test sets) and CORD validation words and lines, where present."""
    out = {}
    try:
        val = datasets.mozhi("val")
        idx = sorted(random.Random(0).sample(range(len(val)), min(mozhi_n, len(val))))
        out["mozhi_val"] = [val[i] for i in idx]
    except (FileNotFoundError, ValueError):
        pass
    if (DATA / "synth" / "val" / "labels.tsv").exists():
        out["synth_val"] = datasets.synth("val")
    folder = M2 / "cord_validation"
    if (folder / "labels.tsv").exists():
        out["cord_val_word"] = datasets.crops(folder, "word")
        out["cord_val_line"] = datasets.crops(folder, "line")
    return out
