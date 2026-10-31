"""M4: every trained field extractor on CORD's 100 test receipts, from true words and from OCR words.

    python scripts/eval_kie.py              results to docs/results/m4_fields.md and .json
    python scripts/eval_kie.py --only smoke runs whose name starts with smoke, results to runs/m4/
    python scripts/eval_kie.py --tiny       the tiny runs of train_kie.py (checks the code), results to runs/m4/

For each run in runs/m4 (its best.pt): field F1 on the test receipts' true words and on their OCR
words (scripts/prepare_m4.py), with a bootstrap 95% interval over receipts. LiLT and LayoutLMv3
are also scored with every word given the same box (the whole receipt), and LayoutLMv3 with a
blank image: how much each leans on layout. Two ceilings: perfect tags on the true words, and on
the OCR words (tags moved from the true words under them), which is what the OCR alone costs.
Runs that differ only in their seed are pooled per receipt for the comparisons; each run's own F1
is listed too.
"""
import argparse
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, kie, stats  # noqa: E402
from doclens.crnn import pick_device  # noqa: E402
from doclens.fields import Receipt, cord_receipts, decode, ocr_receipts, transfer  # noqa: E402

RUNS = REPO / "runs" / "m4"
OUT = REPO / "docs" / "results"


def per_category(true: list[tuple[str, str]], pred: list[tuple[str, str]]) -> dict[str, tuple[int, int, int]]:
    t = Counter((c, "".join(x.split())) for c, x in true)
    p = Counter((c, "".join(x.split())) for c, x in pred)
    out = {}
    for c in {c for c, _ in t} | {c for c, _ in p}:
        tc = Counter({k: v for k, v in t.items() if k[0] == c})
        pc = Counter({k: v for k, v in p.items() if k[0] == c})
        tp = sum((tc & pc).values())
        out[c] = (tp, sum(pc.values()) - tp, sum(tc.values()) - tp)
    return out


def score(receipts: list[Receipt], truth: dict[str, Receipt], tags: dict[str, list[str]]) -> dict:
    """Per receipt TP, FP, FN against the true fields, and per category totals."""
    rows, cats = [], defaultdict(lambda: [0, 0, 0])
    for r in receipts:
        true, pred = truth[r.id].fields(), decode(r.words, tags[r.id])
        pc = per_category(true, pred)
        rows.append([sum(v[i] for v in pc.values()) for i in range(3)])
        for c, v in pc.items():
            for i in range(3):
                cats[c][i] += v[i]
    return {"counts": rows, "categories": dict(cats)}


def same_box(receipts: list[Receipt]) -> list[list[list[int]]]:
    return [[[0, 0, 1000, 1000]] * len(r.words) for r in receipts]


def model_for(run: Path, ckpt: dict, device):
    kind, labels = ckpt["kind"], ckpt["labels"]
    if ckpt["args"].get("tiny"):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(run / "tokenizer")
        model = kie.build(kind, labels, config=kie.tiny_config(kind, len(tok)))
    else:
        from transformers import AutoConfig
        tok = kie.tokenizer(ckpt["tokenizer"])
        model = kie.build(kind, labels, config=AutoConfig.from_pretrained(kie.MODELS[kind]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), tok


def tag(model, kind, tok, labels, receipts, device, boxes=None, blank=False):
    ids = {t: i for i, t in enumerate(labels)}
    ex = []
    for k, r in enumerate(receipts):
        rr = r if not blank else Receipt(r.id, r.words, r.quads, r.size, r.tags, r.groups,
                                         np.full_like(r.image, 255))
        ex += kie.encode(tok, rr, ids, boxes[k] if boxes is not None else None)
    return kie.predict(model, kind, receipts, ex, labels, device)


def evaluate_run(run: Path, truths: list[Receipt], ocr: list[Receipt] | None, device) -> dict:
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    kind, labels, a = ckpt["kind"], ckpt["labels"], ckpt["args"]
    model, tok = model_for(run, ckpt, device)
    truth = {r.id: r for r in truths}
    res: dict = {"run": run.name, "kind": kind, "boxes": a["boxes"], "train_on": a["train_on"], "seed": a["seed"],
                 "val_f1": ckpt["best"]["val_f1"], "step": ckpt["best"]["step"], "inputs": {}}
    segment = a["boxes"] == "segment"
    boxes = [kie.segment_boxes(r) for r in truths] if segment else None
    res["inputs"]["true words"] = score(truths, truth, tag(model, kind, tok, labels, truths, device, boxes))
    if not segment:
        if ocr is not None:
            res["inputs"]["OCR words"] = score(ocr, truth, tag(model, kind, tok, labels, ocr, device))
        if kind in ("lilt", "layoutlmv3"):
            res["inputs"]["true words, one box for all"] = score(
                truths, truth, tag(model, kind, tok, labels, truths, device, same_box(truths)))
        if kind == "layoutlmv3":
            res["inputs"]["true words, blank image"] = score(
                truths, truth, tag(model, kind, tok, labels, truths, device, blank=True))
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return res


def setting(r: dict) -> str:
    reads = "text only" if r["kind"] == "roberta" else f"{r['boxes']} boxes"
    return f"{r['kind']} ({reads}, trained on {r['train_on']} words)"


def pooled(runs: list[dict], name: str) -> list[list[int]]:
    """Per receipt counts summed over the runs (seeds) of one setting."""
    rows = [np.array(r["inputs"][name]["counts"]) for r in runs if name in r["inputs"]]
    return np.sum(rows, axis=0).tolist() if rows else []


def f1_of(rows) -> dict:
    a = np.asarray(rows)
    return stats.bootstrap_f1(a[:, 0], a[:, 1], a[:, 2])


def compare(a_rows, b_rows) -> dict:
    a, b = np.asarray(a_rows), np.asarray(b_rows)
    return stats.paired_f1((a[:, 0], a[:, 1], a[:, 2]), (b[:, 0], b[:, 1], b[:, 2]))


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown(res: dict) -> str:
    L = ["# M4 results: fields on receipts\n",
         f"{res['machine']}. Field F1 in percent on CORD's {res['receipts']} test receipts: a field counts when its"
         " category and its text (compared without spaces) are both right. Bootstrap 95% intervals over receipts; runs"
         " that differ only in their seed are pooled per receipt.\n",
         "## Ceilings\n", "| Input | Field F1 |", "|---|---:|"]
    for name, r in res["ceilings"].items():
        L.append(f"| {name} | {pct(r['f1'])} ({pct(r['lo'])} to {pct(r['hi'])}) |")
    L += ["", "## Models\n", "| Setting | Input | Runs | Field F1 | Each run | Validation F1 |", "|---|---|---:|---:|---|---|"]
    for s in res["settings"]:
        for name, r in s["inputs"].items():
            each = ", ".join(pct(x) for x in r["runs"])
            val = ", ".join(pct(x) for x in s["val_f1"])
            L.append(f"| {s['setting']} | {name} | {len(r['runs'])} | {pct(r['f1'])} ({pct(r['lo'])} to {pct(r['hi'])})"
                     f" | {each} | {val} |")
    if res["comparisons"]:
        L += ["", "## Comparisons (paired over receipts)\n", "| A | B | Input | A minus B | p |", "|---|---|---|---:|---:|"]
        for c in res["comparisons"]:
            p = "< 0.001" if c["p"] < 0.001 else f"{c['p']:.3f}"
            L.append(f"| {c['a']} | {c['b']} | {c['input']} | {pct(c['diff'])} ({pct(c['lo'])} to {pct(c['hi'])}) | {p} |")
    if res["categories"]:
        names = list(res["categories"])
        cats = sorted({c for n in names for c in res["categories"][n]}, key=lambda c: -res["support"].get(c, 0))
        L += ["", "## Field F1 by category (true words; fields in the test truth in brackets)\n",
              "| Category | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for c in cats:
            cells = []
            for n in names:
                v = res["categories"][n].get(c)
                cells.append(pct(2 * v[0] / max(2 * v[0] + v[1] + v[2], 1)) if v else "")
            L.append(f"| {c} ({res['support'].get(c, 0)}) | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="score the tiny runs only")
    ap.add_argument("--only", default=None, help="score only the runs whose name starts with this")
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    runs = sorted(p.parent for p in RUNS.glob("*/best.pt"))
    if args.tiny:
        runs = [r for r in runs if r.name.endswith("_tiny")]
    elif args.only:
        runs = [r for r in runs if r.name.startswith(args.only)]
    else:                                                  # the real runs: no tiny or smoke runs
        runs = [r for r in runs if not r.name.endswith("_tiny") and not r.name.startswith("smoke")]
    if not runs:
        raise SystemExit(f"no runs in {RUNS}: train with scripts/train_kie.py first")
    truths = cord_receipts(args.split, image="small")
    ocr = None if args.no_ocr else ocr_receipts(args.split, truths)
    truth = {r.id: r for r in truths}
    res: dict = {"machine": f"torch {torch.__version__}, {platform.platform()}, {device}", "receipts": len(truths),
                 "ceilings": {}, "settings": [], "comparisons": [], "categories": {}, "support": {}}
    perfect = score(truths, truth, {r.id: r.tags for r in truths})
    res["ceilings"]["true words, perfect tags"] = f1_of(perfect["counts"])
    res["support"] = {c: v[0] + v[2] for c, v in perfect["categories"].items()}
    if ocr is not None:
        moved = [transfer(o, t) for o, t in zip(ocr, truths, strict=True)]
        res["ceilings"]["OCR words, tags of the true words under them"] = f1_of(
            score(moved, truth, {r.id: r.tags for r in moved})["counts"])
    scored = []
    for run in runs:
        print(f"{run.name} ...", flush=True)
        scored.append(evaluate_run(run, truths, ocr, device))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        groups[setting(r)].append(r)
    for name, rs in groups.items():
        s = {"setting": name, "kind": rs[0]["kind"], "boxes": rs[0]["boxes"], "train_on": rs[0]["train_on"],
             "runs": [r["run"] for r in rs], "val_f1": [r["val_f1"] for r in rs], "inputs": {}}
        for inp in rs[0]["inputs"]:
            rows = pooled(rs, inp)
            s["inputs"][inp] = {**f1_of(rows), "runs": [f1_of(r["inputs"][inp]["counts"])["f1"] for r in rs], "counts": rows}
        res["settings"].append(s)
        cats: dict = defaultdict(lambda: [0, 0, 0])
        for r in rs:
            for c, v in r["inputs"]["true words"]["categories"].items():
                for i in range(3):
                    cats[c][i] += v[i]
        res["categories"][name.split(" (")[0] + (" (segment)" if rs[0]["boxes"] == "segment" else "")
                          + (" (OCR trained)" if rs[0]["train_on"] == "ocr" else "")] = dict(cats)
    base = {(s["kind"], s["boxes"], s["train_on"]): s for s in res["settings"]}
    pairs = [("lilt", "roberta"), ("layoutlmv3", "lilt"), ("layoutlmv3", "roberta")]
    for inp in ("true words", "OCR words"):
        for a, b in pairs:
            sa, sb = base.get((a, "word", "true")), base.get((b, "word", "true"))
            if sa and sb and inp in sa["inputs"] and inp in sb["inputs"]:
                res["comparisons"].append({"a": sa["setting"], "b": sb["setting"], "input": inp,
                                           **compare(sa["inputs"][inp]["counts"], sb["inputs"][inp]["counts"])})
    for s in res["settings"]:
        ins = s["inputs"]
        if "OCR words" in ins:
            res["comparisons"].append({"a": s["setting"], "b": s["setting"], "input": "OCR words against true words",
                                       **compare(ins["OCR words"]["counts"], ins["true words"]["counts"])})
        for other in ("true words, one box for all", "true words, blank image"):
            if other in ins:
                res["comparisons"].append({"a": s["setting"], "b": s["setting"], "input": f"{other} against true words",
                                           **compare(ins[other]["counts"], ins["true words"]["counts"])})
        if s["train_on"] == "ocr" and "OCR words" in ins:
            t = base.get((s["kind"], s["boxes"], "true"))
            if t and "OCR words" in t["inputs"]:
                res["comparisons"].append({"a": s["setting"], "b": t["setting"], "input": "OCR words",
                                           **compare(ins["OCR words"]["counts"], t["inputs"]["OCR words"]["counts"])})
    stem = "m4_fields_tiny" if args.tiny else f"m4_fields_{args.only}" if args.only else "m4_fields"
    out = RUNS if args.tiny or args.only else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    (out / f"{stem}.md").write_text(markdown(res), encoding="utf8")
    for s in res["settings"]:
        for inp, r in s["inputs"].items():
            print(f"   {s['setting']:55s} {inp:32s} field F1 {pct(r['f1'])}")
    print(f"wrote {out / stem}.json and .md")


if __name__ == "__main__":
    main()
