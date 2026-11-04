"""M6: do a receipt's sums find, and fix, the fields the OCR misread?

    python scripts/eval_checks.py             results to docs/results/m6_checks.md and .json

On CORD's 100 test receipts: how often each rule of doclens.checks holds on the true fields. Then
on fields read from the OCR words (docTR's detector and the CRNN, as M4 read them), first with
perfect tags (the true fields moved onto the OCR words: only the reading is wrong), then as LiLT
finds them (M4, trained on true words, its three seeds): how often a failing rule points at a
receipt with a misread or lost amount, how many receipts with one it points at, how many one
character fixes are made and how many are right, and field F1 before and after the fixes, with a
paired bootstrap over receipts (seeds pooled per receipt).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doclens import REPO, stats  # noqa: E402
from doclens.checks import fix, rules  # noqa: E402
from doclens.fields import cord_receipts, decode, ocr_receipts, transfer  # noqa: E402

OUT = REPO / "docs" / "results"
CHECKED = {"menu.price", "menu.sub.price", "menu.discountprice", "sub_total.subtotal_price", "sub_total.tax_price",
           "sub_total.service_price", "sub_total.discount_price", "total.total_price", "total.cashprice",
           "total.changeprice"}                          # the amounts the rules use


def key(f):
    return f[0], "".join(f[1].split())


def counts(true, pred) -> list[int]:
    t, p = Counter(map(key, true)), Counter(map(key, pred))
    tp = sum((t & p).values())
    return [tp, sum(p.values()) - tp, sum(t.values()) - tp]


def amount_wrong(true, pred) -> bool:
    """A checked amount read wrong, or lost."""
    t, p = Counter(key(f) for f in true if f[0] in CHECKED), Counter(key(f) for f in pred if f[0] in CHECKED)
    return t != p


def evaluate(pairs: list[tuple[str, list, list]]) -> dict:
    """pairs: (receipt id, true fields, fields found), one per receipt and run (seed)."""
    by_receipt_b: dict[str, list] = {}
    by_receipt_a: dict[str, list] = {}
    flagged = flagged_wrong = wrong = wrong_flagged = fixes = right = 0
    examples = []
    for rid, true, pred in pairs:
        fixed, changes = fix(pred)
        for store, found in ((by_receipt_b, pred), (by_receipt_a, fixed)):
            c = counts(true, found)
            store[rid] = [x + y for x, y in zip(store.get(rid, [0, 0, 0]), c)]
        bad = amount_wrong(true, pred)
        fails = any(not r.holds for r in rules(pred))
        flagged += fails
        flagged_wrong += fails and bad
        wrong += bad
        wrong_flagged += bad and fails
        truth_keys = {key(f) for f in true}
        for c in changes:
            fixes += 1
            ok = key((c["field"], c["to"])) in truth_keys
            right += ok
            if len(examples) < 12:
                examples.append({**c, "right": ok, "receipt": rid})
    ids = sorted(by_receipt_b)
    b, a = np.asarray([by_receipt_b[i] for i in ids]), np.asarray([by_receipt_a[i] for i in ids])
    return {"flagged": flagged, "flagged_with_a_wrong_amount": flagged_wrong, "with_a_wrong_amount": wrong,
            "of_those_flagged": wrong_flagged, "fixes": fixes, "fixes_right": right, "examples": examples,
            "f1_before": stats.bootstrap_f1(b[:, 0], b[:, 1], b[:, 2]),
            "f1_after": stats.bootstrap_f1(a[:, 0], a[:, 1], a[:, 2]),
            "after_vs_before": stats.paired_f1((a[:, 0], a[:, 1], a[:, 2]), (b[:, 0], b[:, 1], b[:, 2]))}


def section(title: str, r: dict, runs: int = 1) -> list[str]:
    each = f" (summed over {runs} seeds)" if runs > 1 else ""
    L = [f"\n## {title}\n",
         f"Receipts flagged, a rule failing on the fields found{each}: {r['flagged']}; of them with a misread or"
         f" lost amount: {r['flagged_with_a_wrong_amount']}. Receipts with a misread or lost amount:"
         f" {r['with_a_wrong_amount']}; of them flagged: {r['of_those_flagged']}.\n",
         f"One character fixes made: {r['fixes']}; right: {r['fixes_right']}. Field F1 before"
         f" {100 * r['f1_before']['f1']:.1f}%, after {100 * r['f1_after']['f1']:.1f}% (difference"
         f" {100 * r['after_vs_before']['diff']:+.2f} points, {100 * r['after_vs_before']['lo']:+.2f} to"
         f" {100 * r['after_vs_before']['hi']:+.2f}, p {r['after_vs_before']['p']:.3f}).\n"]
    if r["examples"]:
        L += ["| Rule | Field | Read | Fixed to | Right |", "|---|---|---|---|---|"]
        L += [f"| {e['rule']} | {e['field']} | {e['from']} | {e['to']} | {'yes' if e['right'] else 'no'} |"
              for e in r["examples"]]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs" / "m4"))
    ap.add_argument("--prefix", default="lilt_word_true_s")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-model", action="store_true", help="only the true fields and perfect tags")
    args = ap.parse_args()
    truths = cord_receipts("test", image="none")
    truth = {r.id: r for r in truths}
    ocr = ocr_receipts("test", truths)

    applies: Counter = Counter()
    holds: Counter = Counter()
    for r in truths:
        for rule in rules(r.fields()):
            applies[rule.name] += 1
            holds[rule.name] += rule.holds
    res: dict = {"receipts": len(ocr), "rules_on_true_fields": {n: {"applies": applies[n], "holds": holds[n]}
                                                                 for n in applies}}
    perfect = []
    for r in ocr:
        moved = transfer(r, truth[r.id])
        perfect.append((r.id, truth[r.id].fields(), decode(moved.words, moved.tags)))
    res["perfect_tags"] = evaluate(perfect)

    runs = []
    if not args.no_model:
        import torch
        from eval_kie import model_for, tag

        from doclens.crnn import pick_device
        device = pick_device(args.device)
        runs = sorted(p.parent for p in Path(args.runs).glob(f"{args.prefix}[0-9]/best.pt"))
        found = []
        for run in runs:
            ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            model, tok = model_for(run, ckpt, device)
            tags = tag(model, ckpt["kind"], tok, ckpt["labels"], ocr, device)
            found += [(r.id, truth[r.id].fields(), decode(r.words, tags[r.id])) for r in ocr]
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        if found:
            res["lilt"] = evaluate(found)
            res["lilt"]["runs"] = [r.name for r in runs]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m6_checks.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    L = ["# M6: a receipt's sums against the OCR's misreadings\n",
         f"CORD's {len(ocr)} test receipts, read as in M4 (docTR's detector, the CRNN).\n",
         "| Rule | Receipts it applies to (true fields) | Holds on the true fields |", "|---|---:|---:|"]
    L += [f"| {k} | {v['applies']} | {v['holds']} |" for k, v in res["rules_on_true_fields"].items()]
    L += section("Perfect tags on the OCR words (only the reading is wrong)", res["perfect_tags"])
    if "lilt" in res:
        L += section("LiLT's fields from the OCR words", res["lilt"], len(runs))
    (OUT / "m6_checks.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
