"""Rescoring the recognizer's candidates with TinyGPT (project 1), a language model of Hindi and Hinglish.

The beam search gives each image a few candidate texts with their CTC log probability: how well
each fits the pixels. TinyGPT adds how likely each one is as text. The pick is

    score = log P_ctc(text | image) + lam * log P_lm(text) + beta * characters

beta rewards length, because every extra token lowers a language model's probability, so without
it the model would prefer dropping characters. lam and beta are chosen on validation data, once
for lines whose best candidate holds Devanagari and once for the rest: TinyGPT read Hindi
Wikipedia and romanised Hinglish, not receipts, so it may help one and not the other.

TinyGPT is a separate package: pip install -e ../TinyGPT (with safetensors). Its model bundle
(models/lm/) is checked against its recorded SHA256 when loaded.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .metrics import levenshtein
from .text import normalize, script_of

LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
BETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


class TinyGPTScorer:
    def __init__(self, bundle: str | Path, device: str | torch.device = "cpu"):
        try:
            from tinygpt.bundle import load_bundle
        except ImportError as e:
            raise ImportError("TinyGPT is not installed: pip install -e ../TinyGPT safetensors") from e
        self.model, self.tok, self.config = load_bundle(bundle, device=device)
        self.device = torch.device(device)
        self.block = int(self.config["model_config"]["block_size"])

    @torch.no_grad()
    def score(self, texts: Sequence[str], batch_size: int = 128) -> list[float]:
        """Natural log probability of each text, read after an end of text token (a fresh document).

        Batches always hold batch_size rows and a multiple of 16 tokens (padding is masked out), so
        the Apple GPU meets a handful of shapes, as in recognize.py."""
        ids = [[self.tok.eot_id] + self.tok.encode(t, allowed_special="none")[: self.block - 1] for t in texts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            n = min(self.block, -(-max(len(ids[i]) for i in idx) // 16) * 16)
            x = torch.full((batch_size, n), self.tok.eot_id, dtype=torch.long)
            mask = torch.zeros((batch_size, n - 1), dtype=torch.bool)
            for j, i in enumerate(idx):
                x[j, : len(ids[i])] = torch.tensor(ids[i])
                mask[j, : len(ids[i]) - 1] = True
            x = x.to(self.device)
            logits, _ = self.model(x)
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(2, x[:, 1:, None]).squeeze(2)
            sums = (logp.cpu() * mask).sum(1)
            for j, i in enumerate(idx):
                out[i] = float(sums[j])
        return out


def group(candidates: list[tuple[str, float]]) -> str:
    """deva when the best candidate has Devanagari in it (Hindi or mixed lines), else latn."""
    return "deva" if candidates and script_of(candidates[0][0]) in ("deva", "mixed") else "latn"


def pick(candidates: list[tuple[str, float]], lm_scores: list[float], lam: float, beta: float) -> str:
    if not candidates:
        return ""
    best = max(range(len(candidates)),
               key=lambda k: candidates[k][1] + lam * lm_scores[k] + beta * len(candidates[k][0]))
    return candidates[best][0]


def tune(refs: list[str], nbest: list[list[tuple[str, float]]], lm: list[list[float]]) -> dict[str, dict]:
    """The (lam, beta) with the fewest character errors on validation data, per group.
    lam = 0 and beta = 0 is the plain beam search, so tuning can only match or beat it here."""
    groups = [group(c) for c in nbest]
    best = {}
    for g in ("deva", "latn"):
        idx = [i for i, gi in enumerate(groups) if gi == g]
        if not idx:
            best[g] = {"lam": 0.0, "beta": 0.0, "n": 0}
            continue
        # every candidate's edit distance to its reference, computed once
        dist = {i: [levenshtein(normalize(refs[i]), normalize(t)) for t, _ in nbest[i]] for i in idx}
        trials = []
        for lam, beta in itertools.product(LAMBDAS, BETAS):
            edits = 0
            for i in idx:
                c = nbest[i]
                if c:
                    k = int(np.argmax([c[j][1] + lam * lm[i][j] + beta * len(c[j][0]) for j in range(len(c))]))
                    edits += dist[i][k]
                else:
                    edits += len(normalize(refs[i]))
            trials.append((edits, lam, beta))
        edits, lam, beta = min(trials, key=lambda t: (t[0], t[1], t[2]))     # ties: the smaller weights
        plain = next(t[0] for t in trials if t[1] == 0.0 and t[2] == 0.0)
        best[g] = {"lam": lam, "beta": beta, "n": len(idx), "edits": edits, "edits_plain_beam": plain}
    return best


def apply(nbest: list[list[tuple[str, float]]], lm: list[list[float]], weights: dict[str, dict]) -> list[str]:
    out = []
    for c, s in zip(nbest, lm, strict=True):
        w = weights[group(c)]
        out.append(pick(c, s, w["lam"], w["beta"]))
    return out
