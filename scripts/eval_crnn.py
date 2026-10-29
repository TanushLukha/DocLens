"""M2: the trained recognizer on M1's test sets, scored with M1's code, against Tesseract.

    python scripts/eval_crnn.py --run main            everything; results to docs/results/m2_crnn.*
    python scripts/eval_crnn.py --run main --quick    200 images per set, to check the pipeline
    python scripts/eval_crnn.py --run synthetic       another run; results to docs/results/m2_crnn_synthetic.*

1. Reads every M1 test set three ways: greedy decoding, beam search (10 candidates), and beam
   search with the candidates rescored by TinyGPT. TinyGPT's two weights are chosen on the
   validation sets (Mozhi validation words, synthetic validation lines, CORD validation), never
   on a test set.
2. Scores with M1's scorer: per set, per damage condition and per script, with a bootstrap 95%
   interval for every CER.
3. Compares with Tesseract's best models, each set in its own language (M1's cached predictions
   in runs/m1/), image by image: the difference in CER, its interval and a p value. The CRNN is
   never told the language.
4. Measures speed on this machine: images per second on the GPU and on the CPU (all cores, as
   Tesseract ran in M1).

Predictions are cached in runs/m2/<run>/preds/ and reused while the checkpoint is unchanged.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, datasets, lm, stats  # noqa: E402
from doclens.degrade import CONDITIONS  # noqa: E402
from doclens.metrics import Scores, align, levenshtein  # noqa: E402
from doclens.recognize import Recognizer  # noqa: E402
from doclens.text import normalize  # noqa: E402
from doclens.traindata import val_sets  # noqa: E402

TEST = {
    "synth_seen": lambda: datasets.synth("test_seen"),
    "synth_unseen": lambda: datasets.synth("test_unseen"),
    "mozhi": lambda: datasets.mozhi("test"),
    "cord_word": lambda: datasets.cord("test", "word"),
    "cord_line": lambda: datasets.cord("test", "line"),
}
DECODERS = ("greedy", "beam", "beam_lm")
M1 = REPO / "runs" / "m1"
OUT = REPO / "docs" / "results"
BEAM = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Cache:
    """One JSON lines file per set and kind (greedy, nbest, lm), keyed by image id."""

    def __init__(self, folder: Path, checkpoint: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        digest, meta = sha256(checkpoint), folder / "meta.json"
        if not meta.exists() or json.loads(meta.read_text()).get("checkpoint") != digest:
            for f in folder.glob("*.jsonl"):
                f.unlink()
            meta.write_text(json.dumps({"checkpoint": digest}))

    def get(self, set_name: str, kind: str, items: list, make):
        path = self.folder / f"{set_name}__{kind}.jsonl"
        if path.exists():
            rows = {r["id"]: r["value"] for r in map(json.loads, path.read_text(encoding="utf8").splitlines())}
            if all(it.meta["id"] in rows for it in items):
                return [rows[it.meta["id"]] for it in items], None
        t = time.time()
        values = make()
        seconds = time.time() - t
        with open(path, "w", encoding="utf8") as f:
            for it, v in zip(items, values, strict=True):
                f.write(json.dumps({"id": it.meta["id"], "value": v}, ensure_ascii=False) + "\n")
        return values, seconds


def edits(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref), normalize(hyp)
    return levenshtein(r, h), len(r)


def summarize(items, hyps) -> dict:
    groups = defaultdict(Scores)
    for it, h in zip(items, hyps, strict=True):
        groups["all"].add(it.text, h)
        for key in ("condition", "script"):
            if key in it.meta:
                groups[f"{key}={it.meta[key]}"].add(it.text, h)
    return {g: s.summary() for g, s in sorted(groups.items())}


def tesseract(set_name: str, items) -> list[str] | None:
    path = M1 / f"{set_name}__best__raw.tsv"
    if not path.exists():
        return None
    rows = {r["id"]: r["hyp"] for r in datasets.read_tsv(path)}
    if not all(it.meta["id"] in rows for it in items):
        return None
    return [rows[it.meta["id"]] for it in items]


def speed(rec: Recognizer, images, n: int = 500) -> float:
    """Images per second for greedy reading of the first n images (after a warm up batch)."""
    sample, r = images[:n], rec
    r.read(sample[:32])
    t = time.perf_counter()
    r.read(sample)
    return len(sample) / (time.perf_counter() - t)       # read() copies results to the CPU: the GPU is done


def confusions(items, hyps, top: int = 15) -> dict:
    sub, dele, ins = Counter(), Counter(), Counter()

    def show(ch):
        return "(space)" if ch == " " else ch

    for it, h in zip(items, hyps, strict=True):
        for op, r, x in align(normalize(it.text), normalize(h)):
            if op == "sub":
                sub[f"{show(r)} > {show(x)}"] += 1
            elif op == "del":
                dele[show(r)] += 1
            elif op == "ins":
                ins[show(x)] += 1
    return {"substituted": sub.most_common(top), "dropped": dele.most_common(top), "invented": ins.most_common(top)}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fewer_more(x: float) -> str:
    """A difference in CER as words, so tables carry no minus signs: "2.1 fewer", "0.4 more"."""
    if abs(x) < 0.0005:
        return "same"
    return f"{pct(abs(x))} {'fewer' if x < 0 else 'more'}"


def markdown(res: dict) -> str:
    L = [f"# M2 results: the CRNN against Tesseract\n\nRun `{res['run']}` (step {res['step']}), {res['params']:,}"
         f" parameters, {res['classes']} classes; {res['machine']}. CER in percent with a bootstrap 95% interval."
         " Tesseract: best models, each set in its own language, page mode 7 (M1). The CRNN reads every set"
         " with one model and is never told the language.\n"]
    L.append("## Headline\n")
    L.append("| Set | Images | Tesseract | CRNN, greedy | CRNN, beam | CRNN, beam + TinyGPT | CRNN against Tesseract | p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s, r in res["sets"].items():
        def ci(k):
            b = r["ci"].get(k)
            return f"{pct(b['rate'])} ({pct(b['lo'])} to {pct(b['hi'])})" if b else ""
        cmp = r.get("vs_tesseract", {}).get(res["main"])
        diff = f"{fewer_more(cmp['diff'])} ({fewer_more(cmp['lo'])} to {fewer_more(cmp['hi'])})" if cmp else ""
        p = ("< 0.001" if cmp["p"] < 0.001 else f"{cmp['p']:.3f}") if cmp else ""
        L.append(f"| {s} | {r['n']} | {ci('tesseract')} | {ci('greedy')} | {ci('beam')} | {ci('beam_lm')} | {diff} | {p} |")
    L.append(f"\nThe comparison column is `{res['main']}` against Tesseract on the same images: how many points of"
             " CER fewer or more the CRNN has, with the interval of that difference.\n")
    L.append("## All measures\n")
    L.append("| Set | Reader | CER | WER | Median CER | Catastrophic | Exact | Wrong script |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s, r in res["sets"].items():
        for k in ("tesseract",) + DECODERS:
            a = r["groups"].get(k, {}).get("all")
            if a:
                L.append(f"| {s} | {k} | {pct(a['cer'])} | {pct(a['wer'])} | {pct(a['median_cer'])} |"
                         f" {pct(a['catastrophic'])} | {pct(a['exact'])} | {pct(a['wrong_script'])} |")
    conds = [c for c in CONDITIONS if any(f"condition={c}" in r["groups"].get("greedy", {})
                                          for r in res["sets"].values())]
    if conds:
        L.append(f"\n## Damage, by condition (synthetic lines, CER; {res['main']} against Tesseract)\n")
        L.append("| Set | Reader | " + " | ".join(conds) + " |")
        L.append("|---|---|" + "---:|" * len(conds))
        for s in ("synth_seen", "synth_unseen"):
            r = res["sets"].get(s)
            if not r:
                continue
            for k in ("tesseract", res["main"]):
                g = r["groups"].get(k, {})
                L.append(f"| {s} | {k} | " + " | ".join(pct(g[f'condition={c}']['cer']) if f"condition={c}" in g else ""
                                                       for c in conds) + " |")
        L.append(f"\n## By script (synthetic lines, CER; {res['main']} against Tesseract)\n")
        L.append("| Set | Reader | deva | latn | mixed |")
        L.append("|---|---|---:|---:|---:|")
        for s in ("synth_seen", "synth_unseen"):
            r = res["sets"].get(s)
            if not r:
                continue
            for k in ("tesseract", res["main"]):
                g = r["groups"].get(k, {})
                L.append(f"| {s} | {k} | " + " | ".join(pct(g[f'script={x}']['cer']) if f"script={x}" in g else ""
                                                       for x in ("deva", "latn", "mixed")) + " |")
    if res.get("lm"):
        L.append("\n## TinyGPT rescoring (weights chosen on validation)\n")
        L.append("| Group | Validation images | lam | beta | Validation edits, plain beam | With TinyGPT |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for g, w in res["lm"]["weights"].items():
            L.append(f"| {g} | {w['n']} | {w['lam']} | {w['beta']} | {w.get('edits_plain_beam', '')} | {w.get('edits', '')} |")
    L.append("\n## Speed on this machine (images per second, greedy decoding)\n")
    L.append("| Set | GPU | CPU, all cores | Tesseract (M1) |")
    L.append("|---|---:|---:|---:|")
    for s, r in res["sets"].items():
        sp = r.get("speed", {})
        cells = [f"{sp[k]:.0f}" if sp.get(k) else "" for k in ("gpu", "cpu", "tesseract")]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("\n## Characters the alphabet lacks (share of each test set's characters)\n")
    L.append("| Set | Missing | Most common |")
    L.append("|---|---:|---|")
    for s, r in res["sets"].items():
        o = r["oov"]
        L.append(f"| {s} | {100 * o['share']:.3f}% | {' '.join(f'{c} ({n})' for c, n in o['top'])} |")
    if res.get("confusions"):
        c = res["confusions"]
        L.append(f"\n## What the CRNN gets wrong in Hindi ({res['main']}: Mozhi plus clean synthetic Devanagari lines)\n")
        L.append("| Substituted (true > read) | Times | Dropped | Times | Invented | Times |")
        L.append("|---|---:|---|---:|---|---:|")
        for i in range(max(len(c["substituted"]), len(c["dropped"]), len(c["invented"]))):
            def cell(lst):
                return f"{lst[i][0]} | {lst[i][1]}" if i < len(lst) else " | "
            L.append(f"| {cell(c['substituted'])} | {cell(c['dropped'])} | {cell(c['invented'])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="200 images per set")
    ap.add_argument("--sets", nargs="*", default=list(TEST))
    ap.add_argument("--lm-bundle", default=str(REPO.parent / "TinyGPT" / "models" / "lm"))
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()
    run_dir = REPO / "runs" / "m2" / args.run
    ckpt = run_dir / args.checkpoint
    if not ckpt.exists():
        sys.exit(f"no {ckpt}: train first (scripts/train_crnn.py --run {args.run})")
    rec = Recognizer.load(ckpt, args.device)
    meta = torch.load(ckpt, map_location="cpu", weights_only=True)
    cache = Cache(run_dir / ("preds_quick" if args.quick else "preds"), ckpt)
    print(f"{args.run}: {ckpt.name}, step {meta.get('step')}, device {rec.device}")

    def load(name, fn):
        try:
            items = fn()
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"SKIP {name}: {e}")
            return None
        return sample(items)

    def sample(items):
        if args.quick and len(items) > 200:
            return [items[i] for i in sorted(random.Random(0).sample(range(len(items)), 200))]
        return items

    tests = {n: x for n, x in ((n, load(n, TEST[n])) for n in args.sets) if x}
    vals = {} if args.no_lm else {n: sample(x) for n, x in val_sets().items()}

    preds, nbests = {}, {}
    for name, items in {**tests, **vals}.items():
        images = [it.image() for it in items]
        g, _ = cache.get(name, "greedy", items, lambda images=images: rec.read(images))
        preds[name] = {"greedy": g}
        nb, _ = cache.get(name, "nbest", items, lambda images=images: rec.nbest(images, BEAM))
        nbests[name] = [[(t, s) for t, s in c] for c in nb]
        preds[name]["beam"] = [c[0][0] if c else "" for c in nbests[name]]
        print(f"read {name}: {len(items)} images")

    lm_info = None
    if not args.no_lm:
        try:
            scorer = lm.TinyGPTScorer(args.lm_bundle, str(rec.device))
        except (ImportError, FileNotFoundError, ValueError) as e:
            scorer = None
            print(f"SKIP TinyGPT rescoring: {e}")
        if scorer is not None:
            lms = {}
            for name, items in {**tests, **vals}.items():
                flat = [t for c in nbests[name] for t, _ in c]

                def make(flat=flat, name=name):
                    scores, out, k = scorer.score(flat), [], 0
                    for c in nbests[name]:
                        out.append(scores[k:k + len(c)])
                        k += len(c)
                    return out
                lms[name], _ = cache.get(name, "lm", items, make)
            refs = [it.text for n in vals for it in vals[n]]
            weights = lm.tune(refs, [c for n in vals for c in nbests[n]], [s for n in vals for s in lms[n]])
            for name in tests:
                preds[name]["beam_lm"] = lm.apply(nbests[name], lms[name], weights)
            lm_info = {"bundle": args.lm_bundle, "weights": weights, "validation": {n: len(v) for n, v in vals.items()}}
            print(f"TinyGPT weights: {weights}")
    main_dec = "beam_lm" if lm_info else "beam"

    abc = rec.alphabet
    cpu_rec = Recognizer(copy.deepcopy(rec.model).cpu(), abc, "cpu")
    res = {"run": args.run, "step": meta.get("step"), "params": sum(p.numel() for p in rec.model.parameters()),
           "classes": len(abc), "main": main_dec, "quick": args.quick, "lm": lm_info,
           "machine": f"torch {torch.__version__}, {platform.platform()}, {rec.device}", "sets": {}}
    for name, items in tests.items():
        r = {"n": len(items), "groups": {}, "ci": {}, "vs_tesseract": {}, "speed": {}}
        hyps = dict(preds[name])
        tess = tesseract(name, items)
        if tess is not None:
            hyps = {"tesseract": tess, **hyps}
            info = json.loads((M1 / f"{name}__best__raw.json").read_text())
            r["speed"]["tesseract"] = info["n"] / info["seconds"] if not args.quick else 0.0
        counts = {k: [edits(it.text, h) for it, h in zip(items, v, strict=True)] for k, v in hyps.items()}
        for k, v in hyps.items():
            r["groups"][k] = summarize(items, v)
            r["ci"][k] = stats.bootstrap_rate([e for e, _ in counts[k]], [c for _, c in counts[k]])
        chars = [c for _, c in counts["greedy"]]
        if tess is not None:
            for k in DECODERS:
                if k in counts:
                    r["vs_tesseract"][k] = stats.paired([e for e, _ in counts[k]], [e for e, _ in counts["tesseract"]], chars)
        if main_dec != "greedy":
            r["main_vs_greedy"] = stats.paired([e for e, _ in counts[main_dec]], [e for e, _ in counts["greedy"]], chars)
        missing = Counter(ch for it in items for ch in abc.missing(normalize(it.text)))
        r["oov"] = {"share": sum(missing.values()) / max(sum(chars), 1), "top": missing.most_common(5)}
        if not args.no_speed:
            images = [it.image() for it in items]
            r["speed"]["gpu"] = speed(rec, images) if rec.device.type != "cpu" else 0.0
            r["speed"]["cpu"] = speed(cpu_rec, images)
        res["sets"][name] = r
        a = r["groups"][main_dec]["all"]
        t = r["groups"].get("tesseract", {}).get("all")
        print(f"{name:12s} {main_dec}: CER {pct(a['cer']):>5s}" + (f"   Tesseract {pct(t['cer']):>5s}" if t else ""))

    sel_items, sel_hyps = [], []
    for name in ("mozhi", "synth_seen"):
        if name in tests:
            for it, h in zip(tests[name], preds[name][main_dec], strict=True):
                if it.meta.get("script") == "deva" and it.meta.get("condition", "clean") == "clean":
                    sel_items.append(it)
                    sel_hyps.append(h)
    res["confusions"] = confusions(sel_items, sel_hyps) if sel_items else {}

    out = run_dir if args.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    stem = "m2_crnn" + ("" if args.run == "main" else f"_{args.run}") + ("_quick" if args.quick else "")
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / f"{stem}.md").write_text(markdown(res), encoding="utf8")
    print(f"wrote {out / stem}.json and .md")


if __name__ == "__main__":
    main()
