"""M5: every system scored the same way, from what its runner wrote to runs/m5/.

    python scripts/m5_score.py            docs/results/m5_compare.md and .json

Only the images every system read are scored, so every number in a table comes from the same
images; a system that failed on an image is charged for it as an empty reading.
Crops: CER per set (Unicode NFC, M1's scorer) with a bootstrap 95% interval over images, and each
system paired with DocLens on the same images. Pages: page CER (M3's scorer: spaces and line
breaks made one space, separator runs left out) in each system's own order; on the synthetic pages
intervals and pairs are over page designs (decision 37), with the page CER by layout and damage.
Receipts: field F1 through M4's extractors, trained on CORD's true words, three seeds pooled per
receipt: RoBERTa on each system's words in the order the system gives them, and LiLT on its words
and their boxes put in the order DocLens reads receipts (docTR's rows), where the system gives boxes
for lines or words; a line's box is shared among its words by their length. Beside them, the best
any tagger could do on those words (the true fields moved onto them, as in M4). Speed from the
time each runner spent per image; memory is each runner's peak, with its server's when it has one.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "m5"))

import common  # noqa: E402

from doclens import REPO, stats  # noqa: E402
from doclens.fields import Receipt, cord_receipts, decode, ocr_receipt, split_line, transfer  # noqa: E402
from doclens.metrics import Scores  # noqa: E402
from doclens.pages import box_quad, order_quad  # noqa: E402
from doclens.pagescore import page_edits  # noqa: E402

OUT = common.RUNS if "M5_RUNS" in os.environ else REPO / "docs" / "results"     # a smoke run keeps its report apart
SYSTEMS = ("doclens", "tesseract", "paddleocr", "easyocr", "paddleocr_vl", "surya")
REFERENCE = "paddleocr_vl_ref"          # a check on PaddleOCR VL, not a system of its own (decision 48)
NAMES = {"doclens": "DocLens", "tesseract": "Tesseract", "paddleocr": "PaddleOCR", "easyocr": "EasyOCR",
         "paddleocr_vl": "PaddleOCR VL 1.6", "surya": "Surya 2"}
LICENCES = {
    "doclens": "MIT (docTR's detector, used for receipts: Apache 2.0)",
    "tesseract": "Apache 2.0",
    "paddleocr": "Apache 2.0 (code and models)",
    "easyocr": "Apache 2.0",
    "paddleocr_vl": "Apache 2.0 (code and the 0.96 billion parameter model)",
    "surya": "code Apache 2.0; weights AI PUBS OpenRAIL M, modified: no use by an organisation above"
             " $5 million of revenue or funding, or in a competing product, except personal use and research",
}
CROP_SETS = {"mozhi": "Mozhi words (Hindi)", "synth": "Synthetic lines, unseen fonts", "cord_word": "CORD words",
             "cord_line": "CORD lines"}
PAGE_SETS = {"synth": "Synthetic pages, unseen fonts", "cord": "CORD receipts"}
EXTRACTORS = {"roberta": "roberta_word_true_s", "lilt": "lilt_word_true_s"}


# reading what the runners wrote
def outputs(system: str, task: str, root: Path | None = None) -> dict[str, dict]:
    path = (root or common.RUNS) / system / f"{task}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def meta(system: str, root: Path | None = None) -> dict:
    path = (root or common.RUNS) / system / "meta.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}


def common_ids(truth: list[dict], outs: dict[str, dict[str, dict]]) -> list[str]:
    """Ids of the images every system with any output for this set has read."""
    have = [o for o in outs.values() if o]
    return [r["id"] for r in truth if all(r["id"] in o for o in have)]


# crops
def crop_counts(truth: list[dict], out: dict[str, dict]) -> tuple[list[int], list[int], dict]:
    sc = Scores()
    edits, chars = [], []
    for r in truth:
        s = sc.add(r["text"], out[r["id"]].get("text", ""))
        edits.append(s.char_edits)
        chars.append(s.chars)
    return edits, chars, sc.summary()


def score_crops(truth: list[dict], outs: dict[str, dict[str, dict]]) -> dict:
    res = {}
    for name in CROP_SETS:
        rows = [r for r in truth if r["set"] == name]
        sys_out = {s: {k: v for k, v in o.items() if v["set"] == name} for s, o in outs.items()}
        ids = set(common_ids(rows, sys_out))
        rows = [r for r in rows if r["id"] in ids]
        if not rows:
            continue
        res[name] = {"n": len(rows), "systems": {}}
        counts = {}
        for s, o in sys_out.items():
            if not o:
                continue
            e, c, summary = crop_counts(rows, o)
            counts[s] = (e, c)
            res[name]["systems"][s] = {**summary, "interval": stats.bootstrap_rate(e, c),
                                       "errors": sum(bool(o[r["id"]].get("error")) for r in rows)}
        if "doclens" in counts:
            for s, (e, c) in counts.items():
                if s != "doclens":
                    res[name]["systems"][s]["vs_doclens"] = stats.paired(e, counts["doclens"][0], c)
    return res


def score_reference(truth: list[dict], served: dict[str, dict], ref: dict[str, dict]) -> dict:
    """PaddleOCR VL's pipeline as it runs on the Mac (its model served by MLX VLM) against its model
    in Transformers, the reference implementation, on the crops both read (decision 48)."""
    res = {}
    for name in CROP_SETS:
        rows = [r for r in truth if r["set"] == name and r["id"] in served and r["id"] in ref]
        if not rows:
            continue
        e1, c1, s1 = crop_counts(rows, served)
        e2, c2, s2 = crop_counts(rows, ref)
        res[name] = {"n": len(rows), "served": {**s1, "interval": stats.bootstrap_rate(e1, c1)},
                     "reference": {**s2, "interval": stats.bootstrap_rate(e2, c2)},
                     "served_vs_reference": stats.paired(e1, e2, c1),
                     "reference_seconds": float(np.mean([ref[r["id"]]["seconds"] for r in rows]))}
    return res


# pages
def design_sums(rows: list[dict], edits: list[int], chars: list[int]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Edits and characters summed per page design (a receipt is a design of its own)."""
    designs = [r["meta"].get("base", r["id"]) for r in rows]
    order = list(dict.fromkeys(designs))
    at = {d: i for i, d in enumerate(order)}
    e, c = np.zeros(len(order)), np.zeros(len(order))
    for d, x, y in zip(designs, edits, chars):
        e[at[d]] += x
        c[at[d]] += y
    return order, e, c


def score_pages(truth: list[dict], outs: dict[str, dict[str, dict]]) -> dict:
    res = {}
    for name in PAGE_SETS:
        rows = [r for r in truth if r["set"] == name]
        sys_out = {s: {k: v for k, v in o.items() if v["set"] == name} for s, o in outs.items()}
        ids = set(common_ids(rows, sys_out))
        rows = [r for r in rows if r["id"] in ids]
        if not rows:
            continue
        res[name] = {"n": len(rows), "designs": len({r["meta"].get("base", r["id"]) for r in rows}), "systems": {}}
        sums = {}
        for s, o in sys_out.items():
            if not o:
                continue
            pairs = [page_edits(r["text"], o[r["id"]].get("text", "")) for r in rows]
            e, c = [p[0] for p in pairs], [p[1] for p in pairs]
            _, de, dc = design_sums(rows, e, c)
            sums[s] = (de, dc)
            groups = defaultdict(lambda: [0, 0])
            for r, x, y in zip(rows, e, c):
                for key in (f"columns {r['meta'].get('columns')}", f"condition {r['meta'].get('condition')}"):
                    groups[key][0] += x
                    groups[key][1] += y
            res[name]["systems"][s] = {
                "page_cer": stats.bootstrap_rate(de, dc), "median": float(np.median(np.divide(e, np.maximum(c, 1)))),
                "over_half": float(np.mean(np.divide(e, np.maximum(c, 1)) > 0.5)),
                "groups": {k: v[0] / max(v[1], 1) for k, v in sorted(groups.items())} if name == "synth" else {},
                "errors": sum(bool(o[r["id"]].get("error")) for r in rows)}
        if "doclens" in sums:
            for s, (de, dc) in sums.items():
                if s != "doclens":
                    res[name]["systems"][s]["vs_doclens"] = stats.paired(de, sums["doclens"][0], dc)
    return res


# fields on receipts
def own_order(rid: str, row: dict, size: tuple[int, int]) -> Receipt:
    """The system's words in the order it gives them; no boxes (RoBERTa reads text only)."""
    words = row.get("text", "").split()
    whole = box_quad(0, 0, size[0], size[1])
    return ocr_receipt(rid, [whole] * len(words), words, size, None)


def boxed_words(row: dict) -> tuple[list[str], list[np.ndarray]] | None:
    """Every word with its box, from a system's lines (a line's box shared among its words), or
    None when the system gives no boxes finer than blocks."""
    if "lines" not in row:
        return None
    words, quads = [], []
    for line in row.get("lines") or []:
        for w, q in split_line(order_quad(line["quad"]), line["text"]):
            words.append(w)
            quads.append(q)
    return words, quads


def rows_receipt(rid: str, row: dict, size: tuple[int, int]) -> Receipt | None:
    from doclens.doctr_lines import rows_order
    found = boxed_words(row)
    if found is None:
        return None
    words, quads = found
    order = rows_order(quads, size)
    return ocr_receipt(rid, [quads[i] for i in order], [words[i] for i in order], size, None)


def ceiling(receipts: list[Receipt], truth: dict[str, Receipt]) -> list[list[int]]:
    from eval_kie import per_category
    rows = []
    for r in receipts:
        moved = transfer(r, truth[r.id])
        pc = per_category(truth[r.id].fields(), decode(moved.words, moved.tags))
        rows.append([sum(v[i] for v in pc.values()) for i in range(3)])
    return rows


def extract(receipts_by_system: dict[str, dict[str, list[Receipt]]], truth: dict[str, Receipt], device,
            runs_root: Path) -> dict[str, dict[str, list[list[int]]]]:
    """Per receipt (TP, FP, FN) of each extractor on each system's receipts, summed over seeds:
    {extractor: {system: rows}}."""
    import torch
    from eval_kie import model_for, score, tag
    out: dict[str, dict[str, list[list[int]]]] = defaultdict(dict)
    for kind, prefix in EXTRACTORS.items():
        runs = sorted(p.parent for p in runs_root.glob(f"{prefix}[0-9]/best.pt"))
        for run in runs:
            ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            model, tok = model_for(run, ckpt, device)
            for s, by_kind in receipts_by_system.items():
                receipts = by_kind.get(kind)
                if not receipts:
                    continue
                counts = score(receipts, truth, tag(model, ckpt["kind"], tok, ckpt["labels"], receipts, device))["counts"]
                prev = out[kind].get(s)
                out[kind][s] = counts if prev is None else (np.asarray(prev) + np.asarray(counts)).tolist()
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        out[kind]["seeds"] = len(runs)
    return out


def score_fields(truth_pages: list[dict], outs: dict[str, dict[str, dict]], spots: dict[str, dict], device,
                 runs_root: Path) -> dict:
    rows = [r for r in truth_pages if r["set"] == "cord"]
    sys_out = {s: {k: v for k, v in o.items() if v["set"] == "cord"} for s, o in outs.items()}
    ids = set(common_ids(rows, sys_out))
    rows = [r for r in rows if r["id"] in ids]
    if not rows:
        return {}
    truth = {r.id: r for r in cord_receipts("test", image="none")}
    rid = {r["id"]: r["id"].split("/", 1)[1] for r in rows}
    built: dict[str, dict[str, list[Receipt]]] = {}
    for s, o in sys_out.items():
        if not o:
            continue
        built[s] = {"roberta": [own_order(rid[r["id"]], o[r["id"]], tuple(r["size"])) for r in rows]}
        source = spots if s == "paddleocr_vl" else o
        boxed = [rows_receipt(rid[r["id"]], source.get(r["id"], {}), tuple(r["size"])) for r in rows]
        if all(b is not None for b in boxed):
            built[s]["lilt"] = boxed
    counts = extract(built, truth, device, runs_root)
    res: dict = {"receipts": len(rows), "seeds": {k: counts[k].pop("seeds", 0) for k in EXTRACTORS}, "systems": {}}
    for s in built:
        entry: dict = {}
        for kind in EXTRACTORS:
            if s in counts[kind]:
                a = np.asarray(counts[kind][s])
                entry[kind] = stats.bootstrap_f1(a[:, 0], a[:, 1], a[:, 2])
                if s != "doclens" and "doclens" in counts[kind]:
                    b = np.asarray(counts[kind]["doclens"])
                    entry[kind]["vs_doclens"] = stats.paired_f1((a[:, 0], a[:, 1], a[:, 2]), (b[:, 0], b[:, 1], b[:, 2]))
        if "lilt" in built[s]:
            c = np.asarray(ceiling(built[s]["lilt"], truth))
            entry["ceiling"] = stats.bootstrap_f1(c[:, 0], c[:, 1], c[:, 2])
        res["systems"][s] = entry
    return res


# speed and memory
def speed(outs: dict[str, dict], truth: list[dict], sets: dict[str, str]) -> dict[str, float]:
    """Mean seconds per image per set, over the images the system read."""
    out = {}
    for name in sets:
        secs = [outs[r["id"]]["seconds"] for r in truth if r["set"] == name and r["id"] in outs]
        if secs:
            out[name] = float(np.mean(secs))
    return out


def peak_gb(task_meta: dict) -> float | None:
    mem = (task_meta.get("memory") or {}).get("peak")
    servers = [v.get("peak") for v in (task_meta.get("servers") or {}).values() if v and v.get("peak")]
    if mem is None:
        return None
    return (mem + sum(servers)) / 1e9


# the report
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def interval(d: dict, key: str = "rate") -> str:
    return f"{pct(d[key])} ({100 * d['lo']:.1f} to {100 * d['hi']:.1f})"


def versus(d: dict | None) -> str:
    if not d:
        return ""
    diff = d.get("diff")
    word = "more" if diff > 0 else "fewer"
    p = "p < 0.001" if d["p"] < 0.001 else f"p {d['p']:.3f}"
    return f"{abs(100 * diff):.1f} {word} ({p})"


def versus_f1(d: dict | None) -> str:
    if not d:
        return ""
    diff = d.get("diff")
    word = "above" if diff > 0 else "below"
    p = "p < 0.001" if d["p"] < 0.001 else f"p {d['p']:.3f}"
    return f"{abs(100 * diff):.1f} {word} ({p})"


def group_label(key: str) -> str:
    kind, _, value = key.partition(" ")
    if kind == "columns":
        return {"1": "one column", "2": "two columns"}.get(value, f"{value} columns")
    return value


def markdown(res: dict) -> str:
    present = [s for s in SYSTEMS if s in res["systems"]]
    L = ["# M5: DocLens against other OCR systems\n",
         "Every system read the same images (scripts/m5_export.py) through its own runner"
         " (scripts/m5/); scripts/m5_score.py scored them. Intervals are bootstrap 95% intervals;"
         " each system is paired with DocLens on the same images (points of CER, or of F1).\n"]
    L += ["## Systems\n", "| System | Version | Runs on | Licence |", "|---|---|---|---|"]
    for s in present:
        m = res["systems"][s]
        L.append(f"| {NAMES[s]} | {m.get('version', '')} | {m.get('device', '')} | {LICENCES[s]} |")
    for name, label in CROP_SETS.items():
        if name not in res["crops"]:
            continue
        r = res["crops"][name]
        L += [f"\n## Crops: {label} ({r['n']} images, CER)\n",
              "| System | CER | Against DocLens | Read perfectly | CER above 50% | Images per second |",
              "|---|---:|---:|---:|---:|---:|"]
        for s in present:
            if s not in r["systems"]:
                continue
            x = r["systems"][s]
            per = res["systems"][s].get("crop_seconds", {}).get(name)
            L.append(f"| {NAMES[s]} | {interval(x['interval'])} | {versus(x.get('vs_doclens'))} | {pct(x['exact'])}"
                     f" | {pct(x['catastrophic'])} | {1 / per:.1f} |" if per else
                     f"| {NAMES[s]} | {interval(x['interval'])} | {versus(x.get('vs_doclens'))} | {pct(x['exact'])}"
                     f" | {pct(x['catastrophic'])} | |")
    for name, label in PAGE_SETS.items():
        if name not in res["pages"]:
            continue
        r = res["pages"][name]
        over = "page designs" if name == "synth" else "receipts"
        L += [f"\n## Pages: {label} ({r['n']} pages, page CER, intervals over {r['designs']} {over})\n",
              "| System | Page CER | Against DocLens | Median page | Pages above 50% | Seconds per page |",
              "|---|---:|---:|---:|---:|---:|"]
        for s in present:
            if s not in r["systems"]:
                continue
            x = r["systems"][s]
            per = res["systems"][s].get("page_seconds", {}).get(name)
            L.append(f"| {NAMES[s]} | {interval(x['page_cer'])} | {versus(x.get('vs_doclens'))} | {pct(x['median'])}"
                     f" | {pct(x['over_half'])} | {per:.2f} |" if per is not None else
                     f"| {NAMES[s]} | {interval(x['page_cer'])} | {versus(x.get('vs_doclens'))} | {pct(x['median'])}"
                     f" | {pct(x['over_half'])} | |")
        if name == "synth":
            keys = sorted({k for s in present if s in r["systems"] for k in r["systems"][s]["groups"]})
            if keys:
                L += ["", "| System | " + " | ".join(group_label(k) for k in keys) + " |",
                      "|---|" + "---:|" * len(keys)]
                for s in present:
                    if s in r["systems"]:
                        g = r["systems"][s]["groups"]
                        L.append(f"| {NAMES[s]} | " + " | ".join(pct(g[k]) if k in g else "" for k in keys) + " |")
    ref = res.get("reference") or {}
    if ref:
        L += ["\n## PaddleOCR VL on the Mac and in its reference implementation\n",
              "The same crops read by PaddleOCR VL's pipeline, its model served by MLX VLM as its Apple silicon guide"
              " sets it up, and by the model in Hugging Face Transformers in float32 on the CPU (decision 48).\n",
              "| Set | Crops | Pipeline, MLX VLM | Transformers, CPU | Pipeline against Transformers | Seconds per crop,"
              " Transformers |", "|---|---:|---:|---:|---:|---:|"]
        for name, x in ref.items():
            L.append(f"| {CROP_SETS[name]} | {x['n']} | {interval(x['served']['interval'])} |"
                     f" {interval(x['reference']['interval'])} | {versus(x['served_vs_reference'])} |"
                     f" {x['reference_seconds']:.2f} |")
    f = res.get("fields") or {}
    if f:
        L += [f"\n## Fields on receipts ({f['receipts']} receipts, field F1, seeds pooled:"
              f" {f['seeds'].get('roberta', 0)} RoBERTa, {f['seeds'].get('lilt', 0)} LiLT)\n",
              "| System | RoBERTa, the system's own order | Against DocLens | LiLT, words and boxes in rows"
              " | Against DocLens | Perfect tags on these words |", "|---|---:|---:|---:|---:|---:|"]
        for s in present:
            if s not in f["systems"]:
                continue
            x = f["systems"][s]
            rob = interval(x["roberta"], "f1") if "roberta" in x else ""
            lil = interval(x["lilt"], "f1") if "lilt" in x else "no boxes for lines"
            ceil = pct(x["ceiling"]["f1"]) if "ceiling" in x else ""
            L.append(f"| {NAMES[s]} | {rob} | {versus_f1(x.get('roberta', {}).get('vs_doclens'))} | {lil}"
                     f" | {versus_f1(x.get('lilt', {}).get('vs_doclens'))} | {ceil} |")
    L += ["\n## Memory and loading\n", "| System | Peak memory, crops | Peak memory, pages | Loading, crops |",
          "|---|---:|---:|---:|"]
    for s in present:
        m = res["systems"][s]
        gb = [m.get(f"{t}_peak_gb") for t in common.TASKS]
        load = m.get("crops_load_seconds")
        L.append(f"| {NAMES[s]} | " + " | ".join(f"{g:.2f} GB" if g else "" for g in gb)
                 + (f" | {load:.0f} s |" if load is not None else " | |"))
    L += ["", "Memory is the peak physical footprint (macOS), which on Apple silicon includes what the GPU holds"
          " for the process, plus the peak of the server a system runs its model in (Surya: llama.cpp; PaddleOCR"
          " VL: MLX VLM). Tesseract's is the largest single process; it runs one per core."]
    return "\n".join(L) + "\n"


def main():
    import argparse

    from doclens.crnn import pick_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="the extractors' device")
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--runs", default=str(REPO / "runs" / "m4"), help="M4's trained extractors")
    args = ap.parse_args()
    truth = {t: common.load(t) for t in common.TASKS}
    outs = {t: {s: outputs(s, t) for s in SYSTEMS} for t in common.TASKS}
    res: dict = {"export": json.loads((common.DATA / "export.json").read_text()) if (common.DATA / "export.json").exists()
                 else {}, "systems": {}}
    for s in SYSTEMS:
        m = meta(s)
        if not m:
            continue
        info: dict = {"machine": m.get("machine", {})}
        for t in common.TASKS:
            tm = m.get("tasks", {}).get(t)
            if not tm:
                continue
            info[f"{t}_peak_gb"] = peak_gb(tm)
            info[f"{t}_load_seconds"] = tm.get("load_seconds")
            info.setdefault("version", tm.get("tag", ""))
            info.setdefault("device", tm.get("device", ""))
            info[f"{t}_meta"] = tm
        info["crop_seconds"] = speed(outs["crops"][s], truth["crops"], CROP_SETS)
        info["page_seconds"] = speed(outs["pages"][s], truth["pages"], PAGE_SETS)
        res["systems"][s] = info
    res["crops"] = score_crops(truth["crops"], outs["crops"])
    res["pages"] = score_pages(truth["pages"], outs["pages"])
    res["reference"] = score_reference(truth["crops"], outs["crops"]["paddleocr_vl"], outputs(REFERENCE, "crops"))
    if not args.no_fields:
        res["fields"] = score_fields(truth["pages"], outs["pages"], outputs("paddleocr_vl", "spotting"),
                                     pick_device(args.device), Path(args.runs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m5_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf8")
    (OUT / "m5_compare.md").write_text(markdown(res), encoding="utf8")
    print(markdown(res))
    print(f"wrote {OUT / 'm5_compare.md'} and .json")


if __name__ == "__main__":
    main()
