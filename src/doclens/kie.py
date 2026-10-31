"""Key information extraction (M4): a transformer tags every word of a receipt with its field.

Three models, compared on the same receipts (decision 39 in docs/PLAN.md):
    roberta      RoBERTa base: the words alone, in reading order.
    lilt         LiLT (RoBERTa base and a layout stream): the words and each word's box.
    layoutlmv3   LayoutLMv3 base: the words, their boxes and the receipt's image.
All three read RoBERTa's byte level BPE vocabulary, so a word becomes the same tokens for each;
only what comes with the words differs. Boxes are scaled to 0 to 1000 across the receipt.

A receipt is cut into windows of whole words that fit 384 tokens (every training receipt fits in
one). Windows are padded to one of three fixed lengths, so the Apple GPU compiles three graphs and
no more (M2, decision 21). A word's tag is read from its first token; the other tokens of a word
are not trained.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .fields import Receipt, counts, decode, f1

MODELS = {
    "roberta": "FacebookAI/roberta-base",
    "lilt": "SCUT-DLVCLab/lilt-roberta-en-base",
    "layoutlmv3": "microsoft/layoutlmv3-base",
}
TOKENIZER = MODELS["roberta"]          # all three read RoBERTa's vocabulary; scripts/bench_m4.py checks it
LENGTHS = (128, 256, 384)              # CORD's longest training receipt is 296 tokens (scripts/bench_m4.py)
IGNORE = -100


@dataclass
class Example:
    receipt: str
    words: range                 # which words of the receipt this window holds
    ids: np.ndarray              # (L,) token ids, padded
    mask: np.ndarray             # (L,) 1 on real tokens
    bbox: np.ndarray             # (L, 4) 0 to 1000
    labels: np.ndarray           # (L,) a label on each word's first token, IGNORE elsewhere
    first: np.ndarray            # (number of words,) position of each word's first token
    pixels: np.ndarray | None    # (224, 224, 3) uint8, for LayoutLMv3
    pad: int = 1                 # the tokenizer's padding id


def box(quad: np.ndarray, size: tuple[int, int]) -> list[int]:
    """A quadrilateral as the upright box around it, scaled to 0 to 1000 across the receipt."""
    w, h = size
    x0, y0 = quad[:, 0].min() / w, quad[:, 1].min() / h
    x1, y1 = quad[:, 0].max() / w, quad[:, 1].max() / h
    b = [int(round(1000 * min(max(v, 0.0), 1.0))) for v in (x0, y0, x1, y1)]
    return [b[0], b[1], max(b[0], b[2]), max(b[1], b[3])]


def segment_boxes(receipt: Receipt) -> list[list[int]]:
    """Every word given its whole field's box, the setting of LayoutLMv3's and LiLT's papers. It
    tells the model where each field starts and ends, which OCR cannot, so it is a check of the
    training code against published numbers, never a result (decision 40)."""
    out = []
    for g in receipt.groups:
        qs = [q for q, h in zip(receipt.quads, receipt.groups, strict=True) if h == g]
        out.append(box(np.concatenate(qs), receipt.size))
    return out


def encode(tokenizer, receipt: Receipt, label_ids: dict[str, int], boxes: list[list[int]] | None = None,
           max_len: int = LENGTHS[-1]) -> list[Example]:
    """A receipt as windows of whole words, each padded to the shortest fixed length it fits."""
    boxes = boxes or [box(q, receipt.size) for q in receipt.quads]
    enc = tokenizer(receipt.words, is_split_into_words=True, add_special_tokens=False)
    per_word: list[list[int]] = [[] for _ in receipt.words]
    for tok, wid in zip(enc["input_ids"], enc.word_ids(), strict=True):
        if wid is not None:
            per_word[wid].append(tok)
    for i, toks in enumerate(per_word):
        if not toks:                                   # a word the tokenizer drops: give it an unknown token
            per_word[i] = [tokenizer.unk_token_id]
    room = max_len - 2
    windows, start, n = [], 0, 0
    for i, toks in enumerate(per_word):
        if n and n + len(toks) > room:
            windows.append(range(start, i))
            start, n = i, 0
        n += min(len(toks), room)
    windows.append(range(start, len(per_word)))
    out = []
    for win in windows:
        ids, bbox, labels, first = [tokenizer.cls_token_id], [[0, 0, 0, 0]], [IGNORE], []
        for i in win:
            toks = per_word[i][:room]
            first.append(len(ids))
            ids += toks
            bbox += [boxes[i]] * len(toks)
            labels += [label_ids.get(receipt.tags[i], 0) if receipt.tags else IGNORE] + [IGNORE] * (len(toks) - 1)
        ids.append(tokenizer.sep_token_id)
        bbox.append([0, 0, 0, 0])
        labels.append(IGNORE)
        length = next(L for L in LENGTHS if L >= len(ids))
        pad = length - len(ids)
        out.append(Example(receipt.id, win, np.array(ids + [tokenizer.pad_token_id] * pad, np.int64),
                           np.array([1] * len(ids) + [0] * pad, np.int64),
                           np.array(bbox + [[0, 0, 0, 0]] * pad, np.int64),
                           np.array(labels + [IGNORE] * pad, np.int64), np.array(first, np.int64), receipt.image,
                           tokenizer.pad_token_id))
    return out


def padded(e: Example, length: int) -> Example:
    """The same window padded further, to a longer fixed length."""
    n = length - len(e.ids)
    if n <= 0:
        return e
    return Example(e.receipt, e.words, np.concatenate([e.ids, np.full(n, e.pad, np.int64)]),
                   np.concatenate([e.mask, np.zeros(n, np.int64)]), np.concatenate([e.bbox, np.zeros((n, 4), np.int64)]),
                   np.concatenate([e.labels, np.full(n, IGNORE, np.int64)]), e.first, e.pixels, e.pad)


def batches(examples: list[Example], size: int, rng: random.Random | None = None, fill: bool = False) -> Iterator[list[Example]]:
    """Batches of one length each, so every batch has one of the fixed shapes. Training (rng given):
    shuffled; the windows left over after a length's full batches are padded up to the next length
    and batched there, so every window is trained on each epoch (only fewer than one batch left at
    the longest length waits for the next epoch). Prediction (fill): in order, the last batch of a
    length filled up with copies of its last window, whose outputs are ignored."""
    by: dict[int, list[Example]] = {}
    for e in examples:
        by.setdefault(len(e.ids), []).append(e)
    out: list[list[Example]] = []
    carry: list[Example] = []
    for length, group in sorted(by.items()):
        group = [padded(e, length) for e in carry] + group
        if rng is not None:
            rng.shuffle(group)
        full = len(group) // size * size
        out += [group[i:i + size] for i in range(0, full, size)]
        rest = group[full:]
        if rng is not None:
            carry = rest
        elif rest:
            out.append(rest + [rest[-1]] * (size - len(rest)) if fill else rest)
    if rng is not None:
        rng.shuffle(out)
    yield from out


def tensors(batch: list[Example], kind: str, device) -> dict:
    import torch
    t = {"input_ids": torch.from_numpy(np.stack([e.ids for e in batch])).to(device),
         "attention_mask": torch.from_numpy(np.stack([e.mask for e in batch])).to(device),
         "labels": torch.from_numpy(np.stack([e.labels for e in batch])).to(device)}
    if kind in ("lilt", "layoutlmv3"):
        t["bbox"] = torch.from_numpy(np.stack([e.bbox for e in batch])).to(device)
    if kind == "layoutlmv3":
        px = np.stack([e.pixels for e in batch]).astype(np.float32) / 255.0      # (B, 224, 224, 3)
        t["pixel_values"] = ((torch.from_numpy(px).permute(0, 3, 1, 2) - 0.5) / 0.5).to(device)
    return t


def build(kind: str, labels: list[str], name: str | None = None, config=None):
    """A token classifier with one output per label: pretrained weights by name, or a fresh model
    from a config (tests)."""
    from transformers import LayoutLMv3ForTokenClassification, LiltForTokenClassification, RobertaForTokenClassification
    cls = {"roberta": RobertaForTokenClassification, "lilt": LiltForTokenClassification,
           "layoutlmv3": LayoutLMv3ForTokenClassification}[kind]
    id2label = dict(enumerate(labels))
    if config is not None:
        config.num_labels, config.id2label, config.label2id = len(labels), id2label, {v: k for k, v in id2label.items()}
        return cls(config)
    return cls.from_pretrained(name or MODELS[kind], num_labels=len(labels), id2label=id2label,
                               label2id={v: k for k, v in id2label.items()})


def tokenizer(name: str = TOKENIZER):
    """RoBERTa's tokenizer, for all three models; add_prefix_space lets it take words."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, add_prefix_space=True)


def predict(model, kind: str, receipts: list[Receipt], examples: list[Example], labels: list[str], device,
            size: int = 8) -> dict[str, list[str]]:
    """Every receipt's words tagged: {receipt id: [tag per word]}."""
    import torch
    by_id = {r.id: r for r in receipts}
    tags: dict[str, list[str]] = {r.id: ["O"] * len(r.words) for r in receipts}
    model.eval()
    seen: set[tuple[str, int]] = set()
    with torch.no_grad():
        for b in batches(examples, size, fill=True):
            logits = model(**{k: v for k, v in tensors(b, kind, device).items() if k != "labels"}).logits
            best = logits.argmax(-1).cpu().numpy()
            for e, row in zip(b, best, strict=True):
                key = (e.receipt, e.words.start)
                if key in seen:                          # a copy that filled up the batch
                    continue
                seen.add(key)
                words = tags[e.receipt]
                for k, i in enumerate(e.words):
                    words[i] = labels[int(row[e.first[k]])]
    assert set(tags) == set(by_id)
    return tags


def field_counts(receipts: list[Receipt], tags: dict[str, list[str]]) -> list[tuple[int, int, int]]:
    """(TP, FP, FN) of every receipt's fields: predicted from the tags against the true fields."""
    return [counts(r.fields(), decode(r.words, tags[r.id])) for r in receipts]


def micro_f1(rows: list[tuple[int, int, int]]) -> float:
    return f1(*(sum(x[i] for x in rows) for i in range(3)))


def train(model, kind: str, examples: list[Example], val: tuple[list[Receipt], list[Example]], labels: list[str],
          device, steps: int, batch_size: int = 8, lr: float = 5e-5, warmup: float = 0.1, eval_every: int = 100,
          seed: int = 0, log: Callable[[dict], None] = print) -> tuple[dict, list[dict]]:
    """AdamW with a linear warmup and decay; the weights with the best validation field F1 are kept.
    Returns those weights (on the CPU) and the validation history."""
    import torch
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=lr)
    warm = max(1, int(warmup * steps))

    def rate(s: int) -> float:                      # up for `warm` steps, then down to zero
        return (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    stream = _forever(examples, batch_size, rng)
    best, best_state, history = -1.0, None, []
    t0, losses = time.time(), []
    for step in range(1, steps + 1):
        model.train()
        out = model(**tensors(next(stream), kind, device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
        if step % eval_every == 0 or step == steps:
            receipts, vex = val
            score = micro_f1(field_counts(receipts, predict(model, kind, receipts, vex, labels, device, batch_size)))
            row = {"step": step, "loss": float(np.mean(losses)), "val_f1": score, "lr": sched.get_last_lr()[0],
                   "seconds": time.time() - t0}
            losses = []
            history.append(row)
            log(row)
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    return best_state, history


def _forever(examples: list[Example], size: int, rng: random.Random) -> Iterator[list[Example]]:
    while True:
        yield from batches(examples, size, rng)


def epochs(examples: list[Example], size: int, steps: int) -> float:
    """How many passes over the training windows `steps` batches make."""
    per = sum(len(g) // size for g in _by_length(examples).values())
    return steps / max(per, 1) if per else math.inf


def _by_length(examples: list[Example]) -> dict[int, list[Example]]:
    by: dict[int, list[Example]] = {}
    for e in examples:
        by.setdefault(len(e.ids), []).append(e)
    return by


# small stand ins, to check the code without downloading anything (tests, --tiny runs)
def tiny_tokenizer(texts: list[str], vocab: int = 1000):
    """A byte level BPE tokenizer trained on the given texts, with RoBERTa's special tokens and ids."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   pad_token="<pad>", cls_token="<s>", sep_token="</s>", mask_token="<mask>")


def tiny_config(kind: str, vocab: int):
    """The architecture of each model at a tiny size: 2 layers of 48 units."""
    import transformers
    common = dict(vocab_size=vocab, hidden_size=48, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                  max_position_embeddings=LENGTHS[-1] + 8, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    if kind == "roberta":
        return transformers.RobertaConfig(**common, type_vocab_size=1)
    if kind == "lilt":
        return transformers.LiltConfig(**common, channel_shrink_ratio=4, max_2d_position_embeddings=1024)
    return transformers.LayoutLMv3Config(**common, coordinate_size=8, shape_size=8, input_size=224, patch_size=16,
                                         max_2d_position_embeddings=1024)
