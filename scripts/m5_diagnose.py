"""M5: why systems failed where they failed, from what their runners wrote (no models needed).

    python scripts/m5_diagnose.py           docs/results/m5_diagnostics.md and .json

Three questions the main tables (scripts/m5_score.py) raise:

1. Two column pages: is a high page CER a misreading or a wrong reading order? Every system that
   boxes its lines or blocks is scored again with its pieces read column after column: left column
   first, each in rows top to bottom and each row left to right, a piece wider than 60% of the page
   counted with the left column (a title at the top, a page number at the foot). This uses where the
   columns are, which only the truth knows, so it is a diagnosis, not a result.
2. Text that is not there: crops whose reading is more than twice as long as the truth and at least
   5 characters longer, and pages read more than one and a half times as long as the truth; with
   the share of the system's edits on that set they cause.
3. Script: crops read partly in another script (M1's measure), and Mozhi's lone dandas.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "m5"))

import common  # noqa: E402

from doclens import REPO  # noqa: E402
from doclens.metrics import Scores, normalize  # noqa: E402
from doclens.pagescore import page_edits  # noqa: E402

SYSTEMS = ("doclens", "tesseract", "paddleocr", "easyocr", "paddleocr_vl", "surya")
NAMES = {"doclens": "DocLens", "tesseract": "Tesseract", "paddleocr": "PaddleOCR", "easyocr": "EasyOCR",
         "paddleocr_vl": "PaddleOCR VL 1.6", "surya": "Surya 2"}
CROP_SETS = ("mozhi", "synth", "cord_word", "cord_line")
OUT = common.RUNS if "M5_RUNS" in os.environ else REPO / "docs" / "results"


def outputs(system: str, task: str) -> dict[str, dict]:
    path = common.RUNS / system / f"{task}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf8").splitlines()) if r}


def pieces(row: dict) -> list[dict] | None:
    """A reading's pieces with boxes, lines or blocks, as (text, quad); None without boxes."""
    found = row.get("lines") if "lines" in row else row.get("blocks")
    if found is None:
        return None
    out = []
    for p in found:
        quad = p.get("quad")
        if quad is None and p.get("box") is not None:                 # PaddleOCR VL: x0, y0, x1, y1
            x0, y0, x1, y1 = p["box"]
            quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if quad is not None and p.get("text"):
            out.append({"text": p["text"], "quad": quad})
    return out


def by_columns(items: list[dict], size: tuple[int, int]) -> str:
    """The pieces read column after column; within a column in rows (docTR's grouping, as DocLens
    orders receipt words), so systems that box words, not lines, keep each line's words in order."""
    import numpy as np

    from doclens.doctr_lines import rows_order
    width = size[0]

    def left(p):
        xs = [q[0] for q in p["quad"]]
        return max(xs) - min(xs) > 0.6 * width or sum(xs) / 4 < width / 2

    out = []
    for side in (True, False):
        part = [p for p in items if left(p) == side]
        order = rows_order([np.float32(p["quad"]) for p in part], size)
        out += [part[i]["text"] for i in order]
    return "\n".join(out)


def columns(truth: list[dict], outs: dict[str, dict]) -> dict:
    rows = [r for r in truth if r["set"] == "synth" and r["meta"].get("columns") == 2]
    res = {}
    for s, o in outs.items():
        if not o or not all(r["id"] in o for r in rows):
            continue
        own = [0, 0]
        cols = [0, 0]
        for r in rows:
            e, c = page_edits(r["text"], o[r["id"]].get("text", ""))
            own[0] += e
            own[1] += c
            ps = pieces(o[r["id"]])
            if ps is None:
                cols = None
                break
            e2, _ = page_edits(r["text"], by_columns(ps, tuple(r["size"])))
            cols[0] += e2
            cols[1] += c
        res[s] = {"pages": len(rows), "own_order": own[0] / own[1]}
        if cols:
            res[s]["by_columns"] = cols[0] / cols[1]
    return res


def invented(truth_crops: list[dict], truth_pages: list[dict], outs_c: dict, outs_p: dict) -> dict:
    res: dict = {"crops": {}, "pages": {}}
    for s in SYSTEMS:
        o = outs_c.get(s) or {}
        for name in CROP_SETS:
            rows = [r for r in truth_crops if r["set"] == name and r["id"] in o]
            if not rows:
                continue
            sc = Scores()
            long_edits = edits = n_long = 0
            for r in rows:
                t, h = normalize(r["text"]), normalize(o[r["id"]].get("text", ""))
                sample = sc.add(r["text"], o[r["id"]].get("text", ""))
                edits += sample.char_edits
                if len(h) > 2 * len(t) and len(h) >= len(t) + 5:
                    n_long += 1
                    long_edits += sample.char_edits
            res["crops"].setdefault(name, {})[s] = {"crops": len(rows), "too_long": n_long,
                                                    "share_of_edits": long_edits / max(edits, 1)}
        o = outs_p.get(s) or {}
        for name in ("synth", "cord"):
            rows = [r for r in truth_pages if r["set"] == name and r["id"] in o]
            if not rows:
                continue
            long = [r["id"] for r in rows if len(normalize(o[r["id"]].get("text", ""))) > 1.5 * len(normalize(r["text"]))]
            res["pages"].setdefault(name, {})[s] = {"pages": len(rows), "too_long": len(long), "ids": long[:10]}
    return res


def script(truth_crops: list[dict], outs_c: dict) -> dict:
    res: dict = {}
    for s in SYSTEMS:
        o = outs_c.get(s) or {}
        for name in ("mozhi", "synth"):
            rows = [r for r in truth_crops if r["set"] == name and r["id"] in o]
            if not rows:
                continue
            sc = Scores()
            for r in rows:
                sc.add(r["text"], o[r["id"]].get("text", ""))
            res.setdefault(name, {})[s] = {"wrong_script": sc.summary()["wrong_script"]}
        dandas = [r for r in truth_crops if r["set"] == "mozhi" and r["text"].strip() == "।" and r["id"] in o]
        if dandas:
            res.setdefault("lone_danda", {})[s] = {"words": len(dandas),
                                                  "misread": sum(normalize(o[r["id"]].get("text", "")) != "।" for r in dandas)}
    return res


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def markdown(res: dict) -> str:
    L = ["# M5: why the systems failed where they did\n",
         "From what each runner wrote; no model is run (scripts/m5_diagnose.py).\n",
         f"## Two column pages ({next(iter(res['columns'].values()))['pages']} pages): order or reading?\n",
         "Page CER as each system gives the page, and with its lines or blocks read column after column (a diagnosis: it"
         " uses where the columns are).\n", "| System | Its own order | Column after column |", "|---|---:|---:|"]
    for s, x in res["columns"].items():
        L.append(f"| {NAMES[s]} | {pct(x['own_order'])} | {pct(x['by_columns']) if 'by_columns' in x else 'no boxes'} |")
    L += ["\n## Text that is not there\n", "Crops read more than twice as long as the truth (and 5 characters longer), and the"
          " share of the system's edits on the set they cause.\n",
          "| System | " + " | ".join(CROP_SETS) + " |", "|---|" + "---:|" * len(CROP_SETS)]
    for s in SYSTEMS:
        cells = []
        for name in CROP_SETS:
            x = res["invented"]["crops"].get(name, {}).get(s)
            cells.append(f"{x['too_long']} ({pct(x['share_of_edits'])})" if x else "")
        L.append(f"| {NAMES[s]} | " + " | ".join(cells) + " |")
    L += ["", "Pages read more than one and a half times as long as the truth:", ""]
    for name, by in res["invented"]["pages"].items():
        L.append(f"* {name}: " + "; ".join(f"{NAMES[s]} {x['too_long']} of {x['pages']}" for s, x in by.items()))
    L += ["\n## Script\n", "| System | Mozhi words partly in another script | Synthetic lines partly in another script |"
          " Mozhi's lone dandas misread |", "|---|---:|---:|---:|"]
    for s in SYSTEMS:
        m, y = res["script"].get("mozhi", {}).get(s), res["script"].get("synth", {}).get(s)
        d = res["script"].get("lone_danda", {}).get(s)
        danda = f"{d['misread']} of {d['words']}" if d else ""
        L.append(f"| {NAMES[s]} | {pct(m['wrong_script']) if m else ''} | {pct(y['wrong_script']) if y else ''} | {danda} |")
    return "\n".join(L) + "\n"


def main():
    truth_c, truth_p = common.load("crops"), common.load("pages")
    outs_c = {s: outputs(s, "crops") for s in SYSTEMS}
    outs_p = {s: outputs(s, "pages") for s in SYSTEMS}
    res = {"columns": columns(truth_p, outs_p), "invented": invented(truth_c, truth_p, outs_c, outs_p),
           "script": script(truth_c, outs_c)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m5_diagnostics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    (OUT / "m5_diagnostics.md").write_text(markdown(res), encoding="utf8")
    print(markdown(res))


if __name__ == "__main__":
    main()
