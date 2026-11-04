"""What every M5 runner shares: the exported test images, timing, memory, and results on disk.

Each OCR system runs in the environment it needs (scripts/setup_m5.sh), through its own runner in
this folder. Every runner reads the same images, exported once by scripts/m5_export.py, and writes
what its system read to runs/m5/<system>/<task>.jsonl, one JSON line per image, so the scorer
(scripts/m5_score.py) sees every system the same way. Only the standard library, numpy and Pillow
are used here, which every environment has.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "m5"
RUNS = Path(os.environ.get("M5_RUNS", REPO / "runs" / "m5"))     # a smoke run writes elsewhere
TASKS = ("crops", "pages")


# the exported images
def load(task: str, root: Path | None = None) -> list[dict]:
    path = (root or DATA) / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/m5_export.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def pick(rows: list[dict], sets: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """The rows of the sets named (all when None), at most `limit` of each set, in file order."""
    out, seen = [], {}
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        if limit is not None and seen.get(r["set"], 0) >= limit:
            continue
        seen[r["set"]] = seen.get(r["set"], 0) + 1
        out.append(r)
    return out


def by_set(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["set"], []).append(r)
    return out


def image(row: dict, mode: str = "L", root: Path | None = None) -> Image.Image:
    """The row's image as Pillow reads it, in mode L (grey) or RGB."""
    with Image.open((root or DATA) / row["file"]) as im:
        return im.convert(mode)


def array(row: dict, mode: str = "L", root: Path | None = None) -> np.ndarray:
    return np.asarray(image(row, mode, root))


# memory
_FIELDS = ("user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
           "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
           "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
           "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
           "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
           "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
           "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
           "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")


class _RusageInfoV4(ctypes.Structure):
    """struct rusage_info_v4 of macOS (bsd/sys/resource.h in Apple's xnu sources)."""
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _FIELDS]


def memory(pid: int | None = None) -> dict | None:
    """A process's memory in bytes, now and at its peak. On macOS its physical footprint, the
    figure Activity Monitor shows as Memory, which on Apple silicon includes what the GPU holds for
    the process; elsewhere its resident size. None when the process cannot be read."""
    pid = pid or os.getpid()
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        info = _RusageInfoV4()
        if lib.proc_pid_rusage(pid, 4, ctypes.byref(info)) != 0:          # 4: RUSAGE_INFO_V4
            return None
        return {"now": int(info.phys_footprint), "peak": int(info.lifetime_max_phys_footprint)}
    status = Path(f"/proc/{pid}/status")
    try:
        lines = status.read_text().splitlines()
    except OSError:
        return None
    vals = {k: int(v.split()[0]) * 1024 for k, _, v in (line.partition(":") for line in lines)
            if k in ("VmRSS", "VmHWM")}
    return {"now": vals.get("VmRSS"), "peak": vals.get("VmHWM")} if vals else None


def children_peak() -> int:
    """The largest peak resident size of any finished child process, in bytes (Tesseract's)."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def pids(name: str) -> list[int]:
    """Running processes of this user with this exact name (a server a system started)."""
    out = subprocess.run(["pgrep", "-x", "-u", str(os.getuid()), name], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()]


# results on disk
class Results:
    """runs/m5/<system>/<task>.jsonl, one line per image read. Lines written with other settings
    (another tag) are dropped when the file is opened, so a changed system is read again, and an
    image already read with these settings is not read twice: a stopped run continues."""

    def __init__(self, system: str, task: str, tag: str, root: Path | None = None):
        self.path = (root or RUNS) / system / f"{task}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if self.path.exists():
            rows = [json.loads(line) for line in self.path.read_text(encoding="utf8").splitlines() if line.strip()]
        keep = [r for r in rows if r.get("tag") == tag]
        if len(keep) != len(rows):
            self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf8")
        self.tag = tag
        self.done = {r["id"] for r in keep}

    def add(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf8") as f:
            f.write(json.dumps({**row, "tag": self.tag}, ensure_ascii=False) + "\n")
        self.done.add(row["id"])


def run(results: Results, rows: list[dict], read: Callable[[list[dict]], list[dict]], batch: int = 1,
        show: int = 0, every: int = 200) -> int:
    """Reads the rows not yet in `results`, `batch` at a time; read(rows) gives one dict per row
    (its text, and its lines when the system finds lines). Each row records its share of the wall
    time of its batch. Prints the first `show` readings, so a smoke run shows what came back."""
    todo = [r for r in rows if r["id"] not in results.done]
    t_all, shown, failed = time.perf_counter(), 0, 0
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        t0 = time.perf_counter()
        try:
            outs = read(part)
        except Exception:                        # the system failed on this batch: each image alone,
            outs = []                            # and an image it cannot read is kept with no text
            for r in part:
                try:
                    outs += read([r])
                    failed = 0
                except Exception as e:
                    failed += 1
                    print(f"   {r['id']}: failed, {type(e).__name__}: {e}"[:300], flush=True)
                    outs.append({"text": "", "error": f"{type(e).__name__}: {e}"[:500]})
                    if failed >= 10:
                        raise RuntimeError("ten images in a row failed: stopping, see the errors above") from e
        share = (time.perf_counter() - t0) / len(part)
        if len(outs) != len(part):
            raise RuntimeError(f"{len(part)} images in, {len(outs)} readings out")
        for r, o in zip(part, outs):
            results.add({"id": r["id"], "set": r["set"], "seconds": share, **o})
            if shown < show:
                shown += 1
                print(f"   {r['id']}: true {r.get('text', '')[:70]!r}")
                print(f"   {' ' * len(r['id'])}  read {o.get('text', '')[:70]!r}", flush=True)
        n = i + len(part)
        if n // every != (n - len(part)) // every or n == len(todo):
            el = time.perf_counter() - t_all
            print(f"   {n} of {len(todo)} read, {el / n:.3f} s each", flush=True)
    return len(todo)


# what ran, and on what
def versions(names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def machine() -> dict:
    info = {"python": platform.python_version(), "system": platform.platform(), "cpu": platform.processor(),
            "cores": os.cpu_count()}
    if sys.platform == "darwin":
        for key, name in (("model", "hw.model"), ("chip", "machdep.cpu.brand_string"), ("ram", "hw.memsize")):
            out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
            info[key] = int(out.stdout) if key == "ram" and out.stdout.strip().isdigit() else out.stdout.strip()
    return info


def _total(entry: dict) -> int:
    """A run's peak memory: its process's, with its servers' (they run at the same time)."""
    own = (entry.get("memory") or {}).get("peak") or 0
    return own + sum((m or {}).get("peak") or 0 for m in (entry.get("servers") or {}).values())


def merged(old: dict | None, new: dict) -> dict:
    """What a task's runs did together. A run with other settings (another tag) replaces the old
    record; a run that read nothing (everything was read before) leaves it as it was; a run that
    went on after a stop adds its images, keeps the memory of whichever run peaked higher (with
    that run's servers), and the loading time of the first run."""
    if not old or old.get("tag") != new.get("tag"):
        return new
    if not new.get("read"):
        return old
    top = new if _total(new) >= _total(old) else old
    out = {**new, "read": old.get("read", 0) + new["read"], "load_seconds": old.get("load_seconds"),
           "memory": top.get("memory"), "runs": old.get("runs", 1) + 1}
    if top.get("servers") is not None:
        out["servers"] = top["servers"]
    else:
        out.pop("servers", None)
    return out


def record(system: str, task: str, info: dict, root: Path | None = None) -> Path:
    """Adds what one runner process did to runs/m5/<system>/meta.json, under its task (see merged)."""
    path = (root or RUNS) / system / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    tasks = meta.setdefault("tasks", {})
    tasks[task] = merged(tasks.get(task), info)
    meta["machine"] = machine()
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf8")
    return path


def args(description: str, extra: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="only these sets (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="at most this many images per set (a smoke run)")
    ap.add_argument("--show", type=int, default=0, help="print the first readings of each set")
    if extra:
        extra(ap)
    return ap.parse_args()


def html_text(html: str) -> str:
    """Plain text from the HTML a VLM writes: a line break for every line or block element, a
    space between table cells, entities decoded, other tags dropped."""
    from html.parser import HTMLParser

    breaks = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "thead", "tbody",
              "ul", "ol", "pre", "caption", "hr"}

    class Text(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in breaks:
                self.parts.append("\n")
            elif tag in ("td", "th"):
                self.parts.append(" ")

        def handle_endtag(self, tag):
            if tag in breaks:
                self.parts.append("\n")

        def handle_data(self, data):
            self.parts.append(data)

    p = Text()
    p.feed(html)
    p.close()
    lines = (" ".join(line.split()) for line in "".join(p.parts).split("\n"))
    return "\n".join(line for line in lines if line)
