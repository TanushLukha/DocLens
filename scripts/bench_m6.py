"""M6: the DocLens service measured end to end, over HTTP, on the Mac's CPU.

    python scripts/bench_m6.py            results to docs/results/m6_service.md and .json

Starts the service (uvicorn, the models in models/ from scripts/export_models.py), runs the end to
end checks of scripts/check_service.py against it, and sends it the images M5 compared every system
on: the 150 synthetic test pages in the held out fonts and CORD's 100 test receipts, one request at
a time. Measured per request: the time the client waits, and the
service's own time for finding, reading and the fields. Then throughput: receipts sent by 1, 2 and
4 clients at once. What came back is scored as M5 scored DocLens: page CER (M3's scorer), and field
F1 on the receipts against CORD's true fields (M4's scorer, a bootstrap interval over receipts),
so the service can be checked against the numbers measured before it existed. Memory is the
service process's peak physical footprint.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "m5"))

import common  # noqa: E402
from check_service import request, run_checks  # noqa: E402

from doclens import REPO, stats  # noqa: E402
from doclens.fields import cord_receipts  # noqa: E402
from doclens.pagescore import page_edits  # noqa: E402

OUT = REPO / "docs" / "results"


def post(url: str, row: dict) -> tuple[float, dict]:
    data = (common.DATA / row["file"]).read_bytes()
    t0 = time.perf_counter()
    code, body = request(url, fields={"image": (Path(row["file"]).name, data, "image/png")})
    wall = time.perf_counter() - t0
    if code != 200:
        raise RuntimeError(f"{row['id']}: the service answered {code}: {body[:200]!r}")
    return wall, json.loads(body)


def percentiles(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"n": len(xs), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def key(f) -> tuple[str, str]:
    return f[0], "".join(f[1].split())


def field_counts(true, found) -> list[int]:
    t, p = Counter(map(key, true)), Counter(map(key, found))
    tp = sum((t & p).values())
    return [tp, sum(p.values()) - tp, sum(t.values()) - tp]


def start(models: Path, port: int, threads: int | None) -> subprocess.Popen:
    env = {**os.environ, "DOCLENS_MODELS": str(models)}
    if threads:
        env["DOCLENS_THREADS"] = str(threads)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "doclens.service:app", "--host", "127.0.0.1", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for _ in range(600):
        try:
            if request(f"http://127.0.0.1:{port}/health")[0] == 200:
                return proc
        except OSError:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"the service stopped: {proc.stderr.read()[-2000:]}")
        time.sleep(0.5)
    proc.kill()
    raise SystemExit("the service did not answer in 5 minutes")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=str(REPO / "models"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=None, help="threads per request (default: every core)")
    ap.add_argument("--clients", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--limit", type=int, default=None, help="at most this many images of each kind (a quick run)")
    a = ap.parse_args()
    rows = common.load("pages")
    pages = [r for r in rows if r["set"] == "synth"][: a.limit]
    receipts = [r for r in rows if r["set"] == "cord"][: a.limit]
    truth = {r.id: r for r in cord_receipts("test", image="none")}
    proc = start(Path(a.models), a.port, a.threads)
    base = f"http://127.0.0.1:{a.port}"
    try:
        health = json.loads(request(base + "/health")[1])
        first = {}
        for kind, row in (("page", pages[0]), ("receipt", receipts[0])):      # loads what loads on first use
            first[kind] = post(f"{base}/{kind}", row)[0]
        run_checks(base)                         # the end to end checks: shapes, refusals, the demo page
        res: dict = {"models": health["models"], "first_request_seconds": first, "kinds": {}}
        for kind, sample in (("page", pages), ("receipt", receipts)):
            walls, server, edits, chars, fc = [], [], [], [], []
            for r in sample:
                wall, body = post(f"{base}/{kind}", r)
                walls.append(wall)
                server.append(body["seconds"])
                e, c = page_edits(r["text"], body["text"])
                edits.append(e)
                chars.append(c)
                if kind == "receipt" and "fields" in body:
                    fc.append(field_counts(truth[r["id"].split("/", 1)[1]].fields(),
                                           [(f["category"], f["text"]) for f in body["fields"]]))
            parts = {p: percentiles([s[p] for s in server]) for p in server[0]}
            entry = {"images": len(sample), "wall": percentiles(walls), "server": parts,
                     "page_cer": float(sum(edits) / max(sum(chars), 1))}
            if kind == "page":
                designs = [r["meta"].get("base", r["id"]) for r in sample]
                order = list(dict.fromkeys(designs))
                e, c = np.zeros(len(order)), np.zeros(len(order))
                for d, x, y in zip(designs, edits, chars):
                    e[order.index(d)] += x
                    c[order.index(d)] += y
                entry["page_cer_interval"] = stats.bootstrap_rate(e, c)
            if fc:
                arr = np.asarray(fc)
                entry["field_f1"] = stats.bootstrap_f1(arr[:, 0], arr[:, 1], arr[:, 2])
            res["kinds"][kind] = entry
            print(f"{kind}: {len(sample)} images, {entry['wall']['p50']:.2f} s median, {entry['wall']['p95']:.2f} s at the"
                  f" 95th percentile; page CER {100 * entry['page_cer']:.1f}%"
                  + (f"; field F1 {100 * entry['field_f1']['f1']:.1f}%" if "field_f1" in entry else ""), flush=True)
        res["throughput"] = {}
        for n in a.clients:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda r: post(f"{base}/receipt", r), receipts))
            el = time.perf_counter() - t0
            res["throughput"][str(n)] = {"receipts": len(receipts), "seconds": el, "per_second": len(receipts) / el}
            print(f"{n} clients at once: {len(receipts) / el:.2f} receipts a second", flush=True)
        res["memory"] = common.memory(proc.pid)
        res["machine"] = common.machine()
        res["threads"] = a.threads
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    res["models_folder_bytes"] = sum(f.stat().st_size for f in Path(a.models).rglob("*") if f.is_file())
    (OUT / "m6_service.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    (OUT / "m6_service.md").write_text(markdown(res), encoding="utf8")
    print(f"wrote {OUT / 'm6_service.md'}")


def markdown(res: dict) -> str:
    m = res["machine"]
    lines = ["# M6: the service end to end\n",
             f"On {m.get('chip', m.get('cpu'))} with {m.get('cores')} cores, the CPU only, over HTTP on the same machine; every"
             f" image sent one at a time unless said. The first page took {res['first_request_seconds']['page']:.1f} s and the"
             f" first receipt {res['first_request_seconds']['receipt']:.1f} s (docTR's detector and LiLT load then).\n",
             "| Kind | Images | Median wait | 95th percentile | Finding | Reading | Fields | Page CER | Field F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, e in res["kinds"].items():
        s = e["server"]
        fields = f"{s['fields']['p50']:.2f} s" if "fields" in s else ""
        f1 = e.get("field_f1")
        f1s = f"{100 * f1['f1']:.1f}% ({100 * f1['lo']:.1f} to {100 * f1['hi']:.1f})" if f1 else ""
        lines.append(f"| {kind} | {e['images']} | {e['wall']['p50']:.2f} s | {e['wall']['p95']:.2f} s |"
                     f" {s['find']['p50']:.2f} s | {s['read']['p50']:.2f} s | {fields} | {100 * e['page_cer']:.1f}% | {f1s} |")
    lines += ["", "Finding, reading and fields are medians of the service's own times; the wait adds HTTP and decoding"
              " the image.",
              "", "| Clients at once | Receipts a second |", "|---:|---:|"]
    lines += [f"| {n} | {t['per_second']:.2f} |" for n, t in res["throughput"].items()]
    mem = res.get("memory") or {}
    if mem.get("peak"):
        lines += ["", f"Peak memory of the service: {mem['peak'] / 2**30:.2f} GB. The models folder: "
                      f"{res['models_folder_bytes'] / 2**20:.0f} MB."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
