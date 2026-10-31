import random

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from doclens import kie  # noqa: E402
from doclens.fields import Receipt, label_list  # noqa: E402
from doclens.pages import box_quad  # noqa: E402

WORDS = ["1", "HOT", "TUNA", "67,000", "2", "ICED", "TEA", "24,000", "SUBTOTAL", "91,000", "TAX", "9,100",
         "TOTAL", "100,100", "CASH", "150,000", "CHANGE", "49,900"]


def tiny_tokenizer():
    return kie.tiny_tokenizer([" ".join(WORDS)] * 20, vocab=320)


def receipt(rid: str, image=True) -> Receipt:
    cats = ["menu.cnt", "menu.nm", "menu.nm", "menu.price", "menu.cnt", "menu.nm", "menu.nm", "menu.price",
            "sub", "sub", "tax", "tax", "total", "total", "cash", "cash", "change", "change"]
    groups, tags, last = [], [], None
    for i, c in enumerate(cats):
        g = i if c != last or c.startswith("menu.cnt") or c == "menu.price" else groups[-1]
        groups.append(g)
        tags.append(("I-" if groups[-1] == (groups[-2] if i else -1) else "B-") + c)
        last = c
    quads = [box_quad(20 + 120 * (i % 4), 40 + 50 * (i // 4), 120 + 120 * (i % 4), 70 + 50 * (i // 4)) for i in range(len(WORDS))]
    img = np.full((224, 224, 3), 255, np.uint8) if image else None
    return Receipt(rid, list(WORDS), quads, (600, 400), tags, groups, img)


def test_box_is_scaled_to_0_1000():
    assert kie.box(box_quad(60, 40, 300, 200), (600, 400)) == [100, 100, 500, 500]
    assert kie.box(box_quad(-5, -5, 700, 500), (600, 400)) == [0, 0, 1000, 1000]


def test_encode_labels_first_tokens_and_pads_to_a_fixed_length():
    tok = tiny_tokenizer()
    r = receipt("0")
    labels = label_list([r])
    ids = {t: i for i, t in enumerate(labels)}
    (e,) = kie.encode(tok, r, ids)
    assert len(e.ids) == 128 and e.mask.sum() == e.ids.tolist().index(tok.pad_token_id)
    assert e.ids[0] == tok.cls_token_id
    assert [labels[e.labels[k]] for k in e.first] == r.tags                  # every word's tag on its first token
    assert (e.labels != kie.IGNORE).sum() == len(r.words)
    assert e.bbox[e.first[3]].tolist() == kie.box(r.quads[3], r.size)
    windows = kie.encode(tok, r, ids, max_len=12)                           # a long receipt becomes windows
    assert len(windows) > 1 and [i for w in windows for i in w.words] == list(range(len(r.words)))


def test_batches_keep_one_shape_and_use_every_window():
    tok = tiny_tokenizer()
    rs = [receipt(str(i)) for i in range(5)]
    labels = label_list(rs)
    ex = [e for r in rs for e in kie.encode(tok, r, {t: i for i, t in enumerate(labels)})]
    train = list(kie.batches(ex, 2, random.Random(0)))
    assert len(train) == 2 and all(len(b) == 2 for b in train)              # one left over at the longest length
    pred = list(kie.batches(ex, 2, fill=True))
    assert len(pred) == 3 and len(pred[-1]) == 2
    long = kie.padded(ex[0], 256)                                           # a longer window: the leftover moves up to it
    assert len(long.ids) == 256 and long.mask.sum() == ex[0].mask.sum() and (long.labels != kie.IGNORE).sum() == len(WORDS)
    mixed = ex[:3] + [long]
    train = list(kie.batches(mixed, 2, random.Random(1)))
    assert sorted(len(b[0].ids) for b in train) == [128, 256]
    assert sorted(id(e) for b in train for e in b if len(e.ids) == 256 and e is long) == [id(long)]
    assert sum(len(b) for b in train) == 4                                  # every window, once


@pytest.mark.parametrize("kind", ["roberta", "lilt", "layoutlmv3"])
def test_each_model_trains_and_tags_every_word(kind):
    import torch
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    rs = [receipt(str(i)) for i in range(4)]
    labels = label_list(rs)
    ids = {t: i for i, t in enumerate(labels)}
    ex = [e for r in rs for e in kie.encode(tok, r, ids)]
    model = kie.build(kind, labels, config=kie.tiny_config(kind, len(tok)))
    rows = []
    state, history = kie.train(model, kind, ex, (rs, ex), labels, "cpu", steps=40, batch_size=2, lr=3e-3,
                               eval_every=20, log=rows.append)
    assert [h["step"] for h in history] == [20, 40] and rows == history
    assert set(state) == set(model.state_dict())
    model.load_state_dict(state)
    tags = kie.predict(model, kind, rs, ex, labels, "cpu", size=2)
    assert set(tags) == {r.id for r in rs} and all(len(tags[r.id]) == len(r.words) for r in rs)
    assert history[-1]["val_f1"] > 0.5                                       # it learns the toy receipts


def test_a_batch_in_parts_takes_the_same_step_as_the_whole_batch():
    import torch
    tok = tiny_tokenizer()
    rs = [receipt(str(i)) for i in range(4)]
    labels = label_list(rs)
    ex = [e for r in rs for e in kie.encode(tok, r, {t: i for i, t in enumerate(labels)})]
    after = []
    for micro in (4, 1):
        torch.manual_seed(0)
        model = kie.build("lilt", labels, config=kie.tiny_config("lilt", len(tok)))
        model.config.hidden_dropout_prob = model.config.attention_probs_dropout_prob = 0.0
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        kie.train(model, "lilt", ex, (rs, ex), labels, "cpu", steps=1, batch_size=4, lr=1e-3, eval_every=1,
                  log=lambda row: None, micro=micro)
        after.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.allclose(after[0], after[1], atol=1e-5)
