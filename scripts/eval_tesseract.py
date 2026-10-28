"""M1: how well Tesseract reads Hindi and English, and what each setting is worth.

    python scripts/eval_tesseract.py            the full study (prints its speed as it goes)
    python scripts/eval_tesseract.py --quick    200 images per set, to check the pipeline

Every set is read in page mode 7 (one line of text: Tesseract finds the text inside the
crop) with the language setting a careful user would pick for it: hin for Mozhi, eng for
CORD, hin+eng for the synthetic lines, which mix both scripts. The experiments:
  margins      Mozhi validation words: how much background around a crop, page mode 7 or 8
  models       Tesseract's fast models against its best models
  languages    -l hin, -l eng and -l hin+eng on every set
  preprocess   every OpenCV pipeline in preprocess.PIPELINES, best models

Predictions are cached in runs/m1/, one TSV per set, models and pipeline, with the exact
configuration beside it: an interrupted run continues where it stopped, and a run whose
configuration changed is done again. Results go to docs/results/m1_tesseract.json and a
readable docs/results/m1_tesseract.md; --quick writes to runs/m1_quick/ instead.
"""
import argparse
import dataclasses
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, REPO, datasets, preprocess, tesseract  # noqa: E402
from doclens.degrade import CONDITIONS  # noqa: E402
from doclens.metrics import Scores, align, levenshtein  # noqa: E402
from doclens.tesseract import TessConfig  # noqa: E402
from doclens.text import normalize  # noqa: E402

RUNS = REPO / "runs" / "m1"
OUT = REPO / "docs" / "results"
PSM = 7
LANG_TAG = {"hin": "hin", "eng": "eng", "hin+eng": "hineng"}


@dataclass(frozen=True)
class EvalSet:
    load: Callable[[], list]
    langs: str        # the setting a careful user would pick for this set
    reader: str       # changes whenever the reader changes, so old predictions are not reused


SETS = {
    "synth_seen": EvalSet(lambda: datasets.synth("test_seen"), "hin+eng", "v1"),
    "synth_unseen": EvalSet(lambda: datasets.synth("test_unseen"), "hin+eng", "v1"),
    "mozhi": EvalSet(lambda: datasets.mozhi("test"), "hin", f"margin {datasets.MOZHI_MARGIN}"),
    "cord_word": EvalSet(lambda: datasets.cord("test", "word"), "eng", "v1"),
    "cord_line": EvalSet(lambda: datasets.cord("test", "line"), "eng", "v1"),
}
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0)
MARGIN_SAMPLE = 2000


def config(models: str, langs: str, psm: int = PSM) -> TessConfig:
    return TessConfig(langs, psm, tessdata=str(DATA / f"tessdata_{models}"))


def config_name(models: str, langs: str, natural: str) -> str:
    """"best" for a set's own language, "best_eng" and so on for the others."""
    return models if langs == natural else f"{models}_{LANG_TAG[langs]}"


def predict(set_name, items, cfg_name, cfg, pipeline, workers, reader):
    """Hypotheses for items, from the cache if this exact run was done before."""
    path = RUNS / f"{set_name}__{cfg_name}__{pipeline}.tsv"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        rows = {r["id"]: r for r in datasets.read_tsv(path)}
        if (info.get("config") == cfg.label() and info.get("reader", "v1") == reader
                and all(it.meta["id"] in rows for it in items)):
            return [rows[it.meta["id"]]["hyp"] for it in items], float(info["seconds"])
    t0 = time.time()
    images = [preprocess.apply(pipeline, it.image()) for it in items]
    hyps = tesseract.recognize(images, cfg, workers)
    seconds = time.time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    datasets.write_tsv(path, ["id", "ref", "hyp"], [[it.meta["id"], it.text, h] for it, h in zip(items, hyps)])
    meta.write_text(json.dumps({"seconds": seconds, "n": len(items), "config": cfg.label(), "reader": reader}))
    return hyps, seconds


def summarize(items, hyps, by=("condition", "script")) -> dict:
    groups = defaultdict(Scores)
    for it, h in zip(items, hyps):
        groups["all"].add(it.text, h)
        for key in by:
            if key in it.meta:
                groups[f"{key}={it.meta[key]}"].add(it.text, h)
    return {g: s.summary() for g, s in sorted(groups.items())}


def confusions(items, hyps, top: int = 15) -> dict:
    sub, dele, ins = Counter(), Counter(), Counter()
    for it, h in zip(items, hyps):
        for op, r, x in align(normalize(it.text), normalize(h)):
            if op == "sub":
                sub[f"{r} > {x}"] += 1
            elif op == "del":
                dele[r] += 1
            elif op == "ins":
                ins[x] += 1
    return {"substituted": sub.most_common(top), "dropped": dele.most_common(top), "invented": ins.most_common(top)}


def fmt(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown(res: dict) -> str:
    L = [f"# M1 results: Tesseract on Hindi and English\n\n{res['tesseract']}; {res['workers']} parallel processes;"
         f" page mode {PSM}. CER and WER in percent (lower is better); median CER per image; catastrophic: share of"
         " images with CER above 50%; exact: share read perfectly; wrong script: share of outputs with letters of a"
         " script the true text does not contain.\n"]
    runs = res["runs"]

    def get(exp, set_name, cfg, pipe="raw"):
        for r in runs:
            if (r["experiment"], r["set"], r["config"], r["pipeline"]) == (exp, set_name, cfg, pipe):
                return r
        return None

    if res.get("margins"):
        m = res["margins"]
        L.append(f"## Crop margin and page mode (Mozhi validation, {m['n']} words, best models, -l hin, CER)\n")
        L.append("Margin: border of paper colour on every side, as a fraction of the crop height.\n")
        L.append("| Margin | Page mode 7: CER | Catastrophic | Page mode 8: CER | Catastrophic |")
        L.append("|---:|---:|---:|---:|---:|")
        for row in m["rows"]:
            L.append(f"| {row['margin']} | {fmt(row['psm7']['cer'])} | {fmt(row['psm7']['catastrophic'])} |"
                     f" {fmt(row['psm8']['cer'])} | {fmt(row['psm8']['catastrophic'])} |")
        L.append("")

    L.append("## Fast against best models (each set's own language, raw images)\n")
    L.append("| Set | Language | Models | Images | CER | WER | Median CER | Catastrophic | Exact | Wrong script | Images per second |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for set_name in SETS:
        for cfg in ("fast", "best"):
            r = get("models", set_name, cfg)
            if r:
                a = r["groups"]["all"]
                L.append(f"| {set_name} | {r['langs']} | {cfg} | {a['n']} | {fmt(a['cer'])} | {fmt(a['wer'])} |"
                         f" {fmt(a['median_cer'])} | {fmt(a['catastrophic'])} | {fmt(a['exact'])} |"
                         f" {fmt(a['wrong_script'])} | {a['n'] / r['seconds']:.0f} |")

    present = {c.split("=")[1] for r in runs if r["set"].startswith("synth") for c in r["groups"] if c.startswith("condition=")}
    conds = [c for c in CONDITIONS if c in present]          # in the order degrade.CONDITIONS lists them
    if conds:
        L.append("\n## Damage, by condition (synthetic lines, best models, hin+eng, raw images, CER)\n")
        L.append("| Set | Script | " + " | ".join(conds) + " |")
        L.append("|---|---|" + "---:|" * len(conds))
        for set_name in ("synth_seen", "synth_unseen"):
            r = get("models", set_name, "best")
            if not r:
                continue
            for script in ("deva", "latn", "mixed"):
                cells = []
                for c in conds:
                    sub = [it for it in r["items"] if it["condition"] == c and it["script"] == script]
                    e, n = sum(i["edits"] for i in sub), sum(i["chars"] for i in sub)
                    cells.append(fmt(e / n) if n else "")
                L.append(f"| {set_name} | {script} | " + " | ".join(cells) + " |")

    L.append("\n## Which language setting (best models, raw images; synthetic sets: clean lines only)\n")
    L.append("| Set | Script | CER, -l hin | CER, -l eng | CER, -l hin+eng | Wrong script, -l hin+eng |")
    L.append("|---|---|---:|---:|---:|---:|")
    for set_name, ev in SETS.items():
        rs = {langs: get("languages", set_name, config_name("best", langs, ev.langs)) for langs in LANG_TAG}
        if not all(rs.values()):
            continue
        keys = [("script=" + s, s) for s in ("deva", "latn", "mixed")] if set_name.startswith("synth") else [("all", "all")]
        for key, label in keys:
            cells = [fmt(rs[l]["groups"].get(key, {}).get("cer", float("nan"))) for l in LANG_TAG]
            ws = rs["hin+eng"]["groups"].get(key, {}).get("wrong_script", float("nan"))
            L.append(f"| {set_name} | {label} | " + " | ".join(cells) + f" | {fmt(ws)} |")

    pipes = list(preprocess.PIPELINES)
    L.append("\n## Cleaning the image first (best models, each set's own language, CER)\n")
    L.append("| Set | Condition | " + " | ".join(pipes) + " |")
    L.append("|---|---|" + "---:|" * len(pipes))
    for set_name in SETS:
        rows = {p: get("preprocess", set_name, "best", p) for p in pipes}
        if not any(rows.values()):
            continue
        keys = ["all"] + (["condition=" + c for c in conds] if set_name.startswith("synth") else [])
        for k in keys:
            cells = [fmt(rows[p]["groups"][k]["cer"]) if rows[p] and k in rows[p]["groups"] else "" for p in pipes]
            L.append(f"| {set_name} | {k.replace('condition=', '')} | " + " | ".join(cells) + " |")

    if res.get("confusions"):
        L.append("\n## What Tesseract gets wrong in Hindi (best models, -l hin: Mozhi; hin+eng: clean synthetic Devanagari lines)\n")
        c = res["confusions"]
        L.append("| Substituted (true > read) | Times | Dropped | Times | Invented | Times |")
        L.append("|---|---:|---|---:|---|---:|")
        for i in range(max(len(c["substituted"]), len(c["dropped"]), len(c["invented"]))):
            cell = lambda lst: (f"{lst[i][0]} | {lst[i][1]}" if i < len(lst) else " | ")
            L.append(f"| {cell(c['substituted'])} | {cell(c['dropped'])} | {cell(c['invented'])} |")
    return "\n".join(L) + "\n"


def margins_study(workers: int) -> dict | None:
    """Mozhi validation words with 0 to 1 times their height of paper around them, page modes 7 and 8."""
    try:
        val = datasets.mozhi("val", margin=0.0)
    except (FileNotFoundError, ValueError) as e:
        print(f"SKIP margins: {e} (python scripts/get_data.py --only mozhi)")
        return None
    sample = [val[i] for i in sorted(random.Random(0).sample(range(len(val)), min(MARGIN_SAMPLE, len(val))))]
    rows = []
    for m in MARGINS:
        items = [dataclasses.replace(it, margin=m) for it in sample]
        row = {"margin": m}
        for psm in (7, 8):
            cfg = config("best", "hin", psm)
            hyps, seconds = predict(f"mozhi_val{len(items)}_margin{int(m * 100):03d}", items, f"best_psm{psm}", cfg, "raw",
                                    workers, "v1")
            row[f"psm{psm}"] = summarize(items, hyps, by=())["all"]
            print(f"margins    mozhi val    margin {m:4.2f} psm {psm}  CER {fmt(row[f'psm{psm}']['cer']):>5s}")
        rows.append(row)
    return {"n": len(sample), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 images per set")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    args = ap.parse_args()
    global RUNS, OUT
    if args.quick:
        RUNS = RUNS.with_name("m1_quick")
        OUT = RUNS
    loaded = {}
    for name in args.sets:
        try:
            items = SETS[name].load()
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"SKIP {name}: {e}")
            continue
        if args.quick and len(items) > 200:
            # a fixed random sample: a regular stride would fall in step with the eight conditions
            items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), 200))]
        loaded[name] = items
        print(f"{name}: {len(items)} images, -l {SETS[name].langs}")
    res = {"tesseract": tesseract.version(), "workers": args.workers, "quick": args.quick, "psm": PSM, "runs": []}
    res["margins"] = None if args.quick else margins_study(args.workers)

    plan = [("models", m, None, "raw") for m in ("fast", "best")]
    plan += [("languages", "best", langs, "raw") for langs in LANG_TAG]
    plan += [("preprocess", "best", None, p) for p in preprocess.PIPELINES]
    for exp, models, langs, pipe in plan:
        for set_name, items in loaded.items():
            ev = SETS[set_name]
            langs_here = langs or ev.langs
            cache_name = set_name
            if exp == "languages" and set_name.startswith("synth"):
                items = [it for it in items if it.meta.get("condition") == "clean"]
                cache_name += "_clean"
            cfg_name = config_name(models, langs_here, ev.langs)
            cfg = config(models, langs_here)
            hyps, seconds = predict(cache_name, items, cfg_name, cfg, pipe, args.workers, ev.reader)
            groups = summarize(items, hyps)
            entry = {"experiment": exp, "set": set_name, "config": cfg_name, "langs": langs_here, "pipeline": pipe,
                     "label": cfg.label(), "reader": ev.reader, "seconds": seconds, "groups": groups}
            if exp == "models" and set_name.startswith("synth"):
                entry["items"] = [{"condition": it.meta["condition"], "script": it.meta["script"],
                                   **dict(zip(("edits", "chars"), _edits(it.text, h)))} for it, h in zip(items, hyps)]
            res["runs"].append(entry)
            a = groups["all"]
            print(f"{exp:10s} {set_name:12s} {cfg_name:11s} {pipe:14s} CER {fmt(a['cer']):>5s}  WER {fmt(a['wer']):>5s}"
                  f"  wrong script {fmt(a['wrong_script']):>5s}  {a['n'] / max(seconds, 1e-9):6.0f} img/s")

    # error analysis on Hindi: Mozhi plus clean synthetic Devanagari lines, best models, own language
    sel_items, sel_hyps = [], []
    for set_name in ("mozhi", "synth_seen"):
        if set_name in loaded:
            ev, items = SETS[set_name], loaded[set_name]
            hyps, _ = predict(set_name, items, "best", config("best", ev.langs), "raw", args.workers, ev.reader)
            for it, h in zip(items, hyps):
                if it.meta.get("script") == "deva" and it.meta.get("condition", "clean") == "clean":
                    sel_items.append(it)
                    sel_hyps.append(h)
    res["confusions"] = confusions(sel_items, sel_hyps) if sel_items else {}

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "m1_tesseract_quick" if args.quick else "m1_tesseract"
    slim = {**res, "runs": [{k: v for k, v in r.items() if k != "items"} for r in res["runs"]]}
    (OUT / f"{stem}.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False))
    (OUT / f"{stem}.md").write_text(markdown(res), encoding="utf8")
    print(f"wrote {OUT / stem}.json and .md")


def _edits(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref), normalize(hyp)
    return levenshtein(r, h), len(r)


if __name__ == "__main__":
    main()
