"""M2: two training runs compared on the same test images, the way eval_crnn.py compares the CRNN
with Tesseract.

    python scripts/compare_runs.py main synthetic       what the real training images add

For each run this rebuilds what eval_crnn.py read (greedy, beam, and beam rescored by TinyGPT with
the run's own weights) from its cached predictions (runs/m2/<run>/preds/) and its results file
(docs/results/m2_crnn*.json), checks that the rebuilt CERs match that file, then compares the two
runs image by image: the difference in CER, its bootstrap 95% interval and a p value, per test
set, per script on the synthetic lines, and per decoder. Writes docs/results/m2_<a>_vs_<b>.md and
.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, datasets, lm, stats  # noqa: E402
from doclens.metrics import levenshtein  # noqa: E402
from doclens.text import normalize  # noqa: E402

OUT = REPO / "docs" / "results"
DECODERS = ("greedy", "beam", "beam_lm")


def results_file(run: str) -> Path:
    return OUT / ("m2_crnn.json" if run == "main" else f"m2_crnn_{run}.json")


def readings(run: str, set_name: str, items, res: dict) -> dict[str, list[str]]:
    folder = REPO / "runs" / "m2" / run / "preds"

    def rows(kind: str) -> list:
        path = folder / f"{set_name}__{kind}.jsonl"
        got = {r["id"]: r["value"] for r in map(json.loads, path.read_text(encoding="utf8").splitlines())}
        return [got[it.meta["id"]] for it in items]

    nbest = [[(t, s) for t, s in c] for c in rows("nbest")]
    out = {"greedy": rows("greedy"), "beam": [c[0][0] if c else "" for c in nbest]}
    if res.get("lm"):
        out["beam_lm"] = lm.apply(nbest, rows("lm"), res["lm"]["weights"])
    return out


def edits(items, hyps) -> tuple[list[int], list[int]]:
    e, c = [], []
    for it, h in zip(items, hyps, strict=True):
        r = normalize(it.text)
        e.append(levenshtein(r, normalize(h)))
        c.append(len(r))
    return e, c


def pct(x: float) -> str:
    return f"{100 * x:.2f}"          # two decimals: some differences are a fraction of a point


def fewer_more(x: float) -> str:
    """A difference in CER as words, so tables carry no minus signs."""
    if abs(x) < 0.00005:
        return "same"
    return f"{pct(abs(x))} {'fewer' if x < 0 else 'more'}"


def cell(cmp: dict) -> str:
    return f"{fewer_more(cmp['diff'])} ({fewer_more(cmp['lo'])} to {fewer_more(cmp['hi'])})"


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the run whose difference is reported (for example main)")
    ap.add_argument("b", help="the run it is compared with (for example synthetic)")
    args = ap.parse_args()
    res = {r: json.loads(results_file(r).read_text(encoding="utf8")) for r in (args.a, args.b)}
    main_dec = "beam_lm" if all(res[r].get("lm") for r in res) else "beam"
    out = {"a": args.a, "b": args.b, "decoder": main_dec, "sets": {}}
    for name, load in datasets.TEST_SETS.items():
        if not all(name in res[r]["sets"] for r in res):
            continue
        items = load()
        hyps = {r: readings(r, name, items, res[r]) for r in res}
        counts = {}
        for r in res:
            for k, v in hyps[r].items():
                e, c = edits(items, v)
                want = res[r]["sets"][name]["groups"][k]["all"]["cer"]
                got = sum(e) / max(sum(c), 1)
                if abs(got - want) > 1e-9:
                    sys.exit(f"{r} {name} {k}: rebuilt CER {got:.6f} but the results file says {want:.6f};"
                             " run eval_crnn.py again")
                counts[(r, k)] = e
        chars = c
        s = {"n": len(items), "decoders": {}, "scripts": {}}
        for k in DECODERS:
            if (args.a, k) in counts and (args.b, k) in counts:
                cmp = stats.paired(counts[(args.a, k)], counts[(args.b, k)], chars)
                s["decoders"][k] = {"a": sum(counts[(args.a, k)]) / sum(chars),
                                    "b": sum(counts[(args.b, k)]) / sum(chars), **cmp}
        if name.startswith("synth"):
            for script in ("deva", "latn", "mixed"):
                idx = [i for i, it in enumerate(items) if it.meta.get("script") == script]
                if idx:
                    ea = [counts[(args.a, main_dec)][i] for i in idx]
                    eb = [counts[(args.b, main_dec)][i] for i in idx]
                    ch = [chars[i] for i in idx]
                    s["scripts"][script] = {"n": len(idx), "a": sum(ea) / sum(ch), "b": sum(eb) / sum(ch),
                                            **stats.paired(ea, eb, ch)}
        out["sets"][name] = s
        d = s["decoders"][main_dec]
        print(f"{name:12s} {args.b} {pct(d['b']):>5s}  {args.a} {pct(d['a']):>5s}  {cell(d)}  p {p_text(d['p'])}")

    a, b = args.a, args.b
    L = [f"# M2: `{a}` against `{b}` on the same test images\n",
         f"Both runs train the same model with the same settings for the same number of steps; `main` draws"
         " synthetic lines, Mozhi training words and CORD training crops (50, 30 and 20% of samples),"
         " `synthetic` draws synthetic lines only. Each reads with the decoder below, its TinyGPT weights"
         f" chosen on validation for that run. CER in percent; `{a}` against `{b}` image by image, with a"
         " bootstrap 95% interval of the difference and a p value.\n",
         f"## Headline ({main_dec})\n",
         f"| Set | Images | {b} | {a} | {a} against {b} | p |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in out["sets"].items():
        d = s["decoders"][main_dec]
        L.append(f"| {name} | {s['n']} | {pct(d['b'])} | {pct(d['a'])} | {cell(d)} | {p_text(d['p'])} |")
    L += [f"\n## By script, synthetic lines ({main_dec})\n",
          f"| Set | Script | Images | {b} | {a} | {a} against {b} | p |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, s in out["sets"].items():
        for script, d in s["scripts"].items():
            L.append(f"| {name} | {script} | {d['n']} | {pct(d['b'])} | {pct(d['a'])} | {cell(d)} | {p_text(d['p'])} |")
    L += ["\n## Every decoder\n", f"| Set | Decoder | {b} | {a} | {a} against {b} | p |",
          "|---|---|---:|---:|---:|---:|"]
    for name, s in out["sets"].items():
        for k, d in s["decoders"].items():
            L.append(f"| {name} | {k} | {pct(d['b'])} | {pct(d['a'])} | {cell(d)} | {p_text(d['p'])} |")
    stem = f"m2_{a}_vs_{b}"
    (OUT / f"{stem}.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    (OUT / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"wrote {OUT / stem}.json and .md")


if __name__ == "__main__":
    main()
