"""M3: whole pages. Line finders and readers on synthetic pages and CORD's receipts.

    python scripts/eval_pages.py              everything; results to docs/results/m3_pages.md and .json
    python scripts/eval_pages.py --quick      2 pages per set, layout and condition: checks the pipeline

Readers, each on every page:
    tesseract          Tesseract's own page analysis and recognition (page mode 3; best models;
                       Hindi and English for synthetic pages, English for receipts)
    tesseract+crnn     Tesseract's lines, read by M2's CRNN
    classical+crnn     DocLens's classical line finder (doclens.layout), read by the CRNN
    doctr+crnn         docTR's pretrained DBNet and its reading order, read by the CRNN
    doctr rows+crnn    the same lines in docTR's default order (top to bottom, left to right)
    true+crnn          the true lines, read by the CRNN: what reading costs without any finding

For each: page CER with a bootstrap 95% interval and a paired comparison with Tesseract, both over
page designs (a design's five damaged pages share text and layout, so they are drawn together;
every receipt is its own design); line
precision, recall and F1 at an IoU of 0.5, splits and merges; Kendall's tau of the reading order;
seconds per page. Line finding and reading are cached in runs/m3/ (reading per CRNN checkpoint);
a cached result is used only for the same page image and the same finder code, and a reading only
for the same lines, so redrawn pages or a changed finder are always run again.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, REPO, doctr_lines, layout, preprocess, stats, tesseract  # noqa: E402
from doclens.crnn import pick_device  # noqa: E402
from doclens.layout import LineFinder  # noqa: E402
from doclens.pagereader import read_page  # noqa: E402
from doclens.pages import Page, cord_pages, read_set  # noqa: E402
from doclens.pagescore import score_page  # noqa: E402
from doclens.recognize import Recognizer  # noqa: E402

RUNS = REPO / "runs" / "m3"
OUT = REPO / "docs" / "results"
SETS = ("test_seen", "test_unseen", "cord")
FINDERS = ("tesseract", "classical", "doctr", "doctr rows", "true")
READERS = ("tesseract", "tesseract+crnn", "classical+crnn", "doctr+crnn", "doctr rows+crnn", "true+crnn")


def load(name: str, quick: bool) -> list[Page]:
    pages = cord_pages("test") if name == "cord" else read_set(name)
    if quick:
        seen: dict[tuple, int] = defaultdict(int)
        keep = []
        for p in pages:
            key = (p.meta.get("script"), p.meta.get("columns"), p.meta.get("condition"))
            if seen[key] < 2:
                seen[key] += 1
                keep.append(p)
        pages = keep
    return pages


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def page_key(p: Page) -> str:
    """A fingerprint of a page's pixels."""
    data = p.path.read_bytes() if p.array is None and p.path is not None else p.image().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def code_key(*modules) -> str:
    """A fingerprint of the code of the modules a finder runs."""
    h = hashlib.sha256()
    for m in modules:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()[:16]


def lines_key(quads: list) -> str:
    return hashlib.sha256(json.dumps(quads).encode("utf8")).hexdigest()[:16]


class Store:
    """One JSON file per set and finder (or reading), keyed by page id; every entry carries the key
    of what it was computed from."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def get(self, name: str) -> dict:
        path = self.folder / f"{name}.json"
        return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}

    def put(self, name: str, data: dict) -> None:
        (self.folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")


def find_lines(set_name: str, pages: list[Page], store: Store, args) -> dict[str, dict]:
    """Every finder's lines on every page: {finder: {page id: {"key", "quads", "seconds"[, "texts"]}}}."""
    out = {}
    pkey = {p.id: page_key(p) for p in pages}

    def stale(cache: dict, code: str) -> list[Page]:
        return [p for p in pages if cache.get(p.id, {}).get("key") != f"{pkey[p.id]} {code}"]

    tess = store.get("tesseract")
    langs = "eng" if set_name == "cord" else "hin+eng"
    cfg = tesseract.TessConfig(langs=langs, psm=3, tessdata=str(DATA / "tessdata_best"))
    code = f"{tesseract.version()} {langs} psm 3 best {code_key(tesseract)}".replace(" ", "_")
    todo = stale(tess, code)
    if todo:
        t = time.perf_counter()
        results = tesseract.read_pages([p.image() for p in todo], cfg, workers=args.workers)
        per = (time.perf_counter() - t) * args.workers / max(len(todo), 1)     # one core's time per page
        for p, r in zip(todo, results, strict=True):
            tess[p.id] = {"key": f"{pkey[p.id]} {code}", "quads": [q.tolist() for q in r.quads], "texts": r.texts,
                          "seconds": per}
        store.put("tesseract", tess)
    out["tesseract"] = tess

    classical = store.get("classical")
    code = code_key(layout, preprocess)
    todo = stale(classical, code)
    finder = LineFinder()
    for p in todo:
        t = time.perf_counter()
        quads = finder(p.image())
        classical[p.id] = {"key": f"{pkey[p.id]} {code}", "quads": [q.tolist() for q in quads],
                           "seconds": time.perf_counter() - t}
    if todo:
        store.put("classical", classical)
    out["classical"] = classical

    order, rows = store.get("doctr"), store.get("doctr rows")
    dcode: str | None
    try:
        dcode = f"doctr_{metadata.version('python-doctr')}_db_resnet50_{code_key(doctr_lines)}"
    except metadata.PackageNotFoundError:
        dcode = None
    todo = stale(order, dcode) if dcode else pages
    if todo and dcode and not args.no_doctr:
        det = doctr_lines.DoctrLines(device=str(pick_device(args.doctr_device)))
        for p in todo:
            t = time.perf_counter()
            found = det(p.image())
            s = time.perf_counter() - t
            key = f"{pkey[p.id]} {dcode}"
            order[p.id] = {"key": key, "quads": [q.tolist() for q in found["order"]], "seconds": s}
            rows[p.id] = {"key": key, "quads": [q.tolist() for q in found["rows"]], "seconds": s}
        store.put("doctr", order)
        store.put("doctr rows", rows)
        todo = []
    if not todo:
        out["doctr"], out["doctr rows"] = order, rows
    else:
        print(f"   docTR left out: {len(todo)} pages have no docTR lines for this image and code", flush=True)
    out["true"] = {p.id: {"quads": [line.quad.tolist() for line in p.lines], "seconds": 0.0} for p in pages}
    return out


def read_lines(pages: list[Page], found: dict[str, dict], store: Store, rec: Recognizer, tag: str) -> dict[str, dict]:
    """The CRNN's reading of every finder's lines: {finder: {page id: {"key", "texts", "seconds"}}}. A
    line another finder already gave on the same page (docTR's two orders hold the same lines) is
    not read twice."""
    out: dict[str, dict] = {}
    for finder, by_page in found.items():
        name = f"read {finder} {tag}"
        done = store.get(name)
        todo = [p for p in pages if p.id in by_page and done.get(p.id, {}).get("key") != lines_key(by_page[p.id]["quads"])]
        for p in todo:
            quads = by_page[p.id]["quads"]
            keys = [json.dumps(q) for q in quads]
            same = next((f for f in out if sorted(keys) == sorted(json.dumps(q) for q in found[f][p.id]["quads"])), None)
            if same is not None:          # the same lines in another order: the same reading, and its time
                text = dict(zip((json.dumps(q) for q in found[same][p.id]["quads"]), out[same][p.id]["texts"], strict=True))
                done[p.id] = {"key": lines_key(quads), "texts": [text[k] for k in keys],
                              "seconds": out[same][p.id]["seconds"]}
                continue
            t = time.perf_counter()
            done[p.id] = {"key": lines_key(quads), "texts": read_page(rec, p.image(), [np.float32(q) for q in quads]),
                          "seconds": time.perf_counter() - t}
        if todo:
            store.put(name, done)
        out[finder] = done
    return out


def score(pages: list[Page], found: dict, read: dict) -> dict[str, list[dict]]:
    """Per reader, one score dict per page (pages in the given order)."""
    out: dict[str, list[dict]] = {}
    for reader in READERS:
        finder = reader.removesuffix("+crnn")
        if finder not in found:
            continue
        rows = []
        for p in pages:
            f = found[finder][p.id]
            text = "\n".join(f["texts"]) if reader == "tesseract" else "\n".join(read[finder][p.id]["texts"])
            quads = None if finder == "true" else [np.float32(q) for q in f["quads"]]
            s = score_page(p.lines, quads, text)
            s["seconds_find"] = f["seconds"]
            s["seconds_read"] = 0.0 if reader == "tesseract" else read[finder][p.id]["seconds"]
            rows.append(s)
        out[reader] = rows
    return out


def summarize(pages: list[Page], scores: dict[str, list[dict]]) -> dict:
    designs = [p.meta.get("base", p.id) for p in pages]           # a receipt is a design of its own
    order = list(dict.fromkeys(designs))                           # in page order

    def per_design(values: list[int]) -> list[int]:
        total: defaultdict[str, int] = defaultdict(int)
        for d, v in zip(designs, values, strict=True):
            total[d] += v
        return [total[d] for d in order]

    res: dict = {"pages": len(pages), "designs": len(order), "readers": {}}
    tess = scores.get("tesseract")
    for reader, rows in scores.items():
        e = [r["edits"] for r in rows]
        c = [r["chars"] for r in rows]
        page_cer = [a / max(b, 1) for a, b in zip(e, c, strict=True)]
        r = {"cer": stats.bootstrap_rate(per_design(e), per_design(c)), "median_page_cer": float(np.median(page_cer)),
             "seconds_find": float(np.mean([x["seconds_find"] for x in rows])),
             "seconds_read": float(np.mean([x["seconds_read"] for x in rows]))}
        if "found_lines" in rows[0]:
            tl, fl, m = (sum(x[k] for x in rows) for k in ("true_lines", "found_lines", "matched"))
            prec, rec = m / max(fl, 1), m / max(tl, 1)
            taus = [x["tau"] for x in rows if x["tau"] is not None]
            r.update(true_lines=tl, found_lines=fl, matched=m, precision=prec, recall=rec,
                     f1=2 * prec * rec / max(prec + rec, 1e-9), splits=sum(x["splits"] for x in rows),
                     merges=sum(x["merges"] for x in rows), tau=float(np.mean(taus)) if taus else None)
        if tess is not None and reader != "tesseract":
            r["vs_tesseract"] = stats.paired(per_design(e), per_design([x["edits"] for x in tess]), per_design(c))
        groups: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p, x in zip(pages, rows, strict=True):
            for key in ("condition", "script", "columns"):
                if key in p.meta:
                    g = groups[f"{key}={p.meta[key]}"]
                    g[0] += x["edits"]
                    g[1] += x["chars"]
        r["groups"] = {k: v[0] / max(v[1], 1) for k, v in sorted(groups.items())}
        res["readers"][reader] = r
    return res


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fewer_more(x: float) -> str:
    if abs(x) < 0.0005:
        return "same"
    return f"{pct(abs(x))} {'fewer' if x < 0 else 'more'}"


def markdown(res: dict) -> str:
    L = ["# M3 results: whole pages\n",
         f"{res['machine']}. Page CER in percent with a bootstrap 95% interval over page designs (a design's five"
         " damaged pages drawn together; on CORD, receipts), and the comparison with Tesseract paired the same way;"
         " lines match at an IoU of"
         " 0.5; tau is Kendall's tau of the reading order of matched lines (1: the true order). Seconds per page"
         " for finding lines and for reading them (Tesseract's time is its whole run, on one core).\n"]
    for set_name, s in res["sets"].items():
        L.append(f"## {set_name} ({s['pages']} pages, {s['designs']} designs)\n")
        L.append("| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for reader, r in s["readers"].items():
            ci = r["cer"]
            cmp = r.get("vs_tesseract")
            vs = f"{fewer_more(cmp['diff'])} ({fewer_more(cmp['lo'])} to {fewer_more(cmp['hi'])})" if cmp else ""
            p = ("< 0.001" if cmp["p"] < 0.001 else f"{cmp['p']:.3f}") if cmp else ""
            if "f1" in r:
                tau = f"{r['tau']:.3f}" if r.get("tau") is not None else ""
                det = [pct(r["precision"]), pct(r["recall"]), pct(r["f1"]), str(r["splits"]), str(r["merges"]), tau]
            else:
                det = [""] * 6
            cells = [reader, f"{pct(ci['rate'])} ({pct(ci['lo'])} to {pct(ci['hi'])})", vs, p, *det,
                     f"{r['seconds_find'] + r['seconds_read']:.2f}"]
            L.append("| " + " | ".join(cells) + " |")
        conds = sorted({k for r in s["readers"].values() for k in r["groups"] if k.startswith("condition=")})
        if len(conds) > 1:
            L.append(f"\n### {set_name}: page CER by condition\n")
            L.append("| Reader | " + " | ".join(c.split("=")[1] for c in conds) + " |")
            L.append("|---|" + "---:|" * len(conds))
            for reader, r in s["readers"].items():
                L.append(f"| {reader} | " + " | ".join(pct(r["groups"].get(c, float("nan"))) for c in conds) + " |")
        other = sorted({k for r in s["readers"].values() for k in r["groups"] if k.startswith(("script=", "columns="))})
        if len(other) > 1:
            L.append(f"\n### {set_name}: page CER by script and columns\n")
            L.append("| Reader | " + " | ".join(o.replace("columns=", "columns ").replace("script=", "") for o in other) + " |")
            L.append("|---|" + "---:|" * len(other))
            for reader, r in s["readers"].items():
                L.append(f"| {reader} | " + " | ".join(pct(r["groups"].get(o, float("nan"))) for o in other) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 pages per set, layout and condition")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--checkpoint", default=str(REPO / "runs" / "m2" / "main" / "best.pt"))
    ap.add_argument("--device", default="auto", help="the CRNN's device")
    ap.add_argument("--doctr-device", default="auto", help="docTR's device: auto (the GPU when there is one), cpu, mps")
    ap.add_argument("--no-doctr", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="Tesseract processes (default: all cores)")
    args = ap.parse_args()
    args.workers = args.workers or __import__("os").cpu_count() or 1
    rec = Recognizer.load(args.checkpoint, args.device)
    tag = sha256(Path(args.checkpoint))
    res = {"machine": f"torch {torch.__version__}, {platform.platform()}, CRNN on {pick_device(args.device)},"
                      f" {tesseract.version()}", "checkpoint": str(args.checkpoint), "sets": {}}
    for name in args.sets:
        pages = load(name, args.quick)
        store = Store(RUNS / ("quick" if args.quick else "full") / name)
        t = time.time()
        found = find_lines(name, pages, store, args)
        print(f"{name}: {len(pages)} pages, lines found by {', '.join(found)} ({time.time() - t:.0f} s)", flush=True)
        read = read_lines(pages, found, store, rec, tag)
        res["sets"][name] = summarize(pages, score(pages, found, read))
        for reader, r in res["sets"][name]["readers"].items():
            extra = f"  F1 {pct(r['f1'])}  tau {r['tau']:.3f}" if r.get("tau") is not None else ""
            print(f"   {reader:16s} page CER {pct(r['cer']['rate']):>5s}{extra}", flush=True)
    stem = "m3_pages_quick" if args.quick else "m3_pages"
    out = RUNS if args.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    (out / f"{stem}.md").write_text(markdown(res), encoding="utf8")
    print(f"wrote {out / stem}.json and .md")


if __name__ == "__main__":
    main()
