"""Downloads the data and models, into data/ (never committed).

    python scripts/get_data.py                              what M1 needs
    python scripts/get_data.py --only mozhi_train cord_train  training data, from M2 on

* mozhi        Mozhi Hindi printed words, test and val splits (IIIT Hyderabad, CC BY 4.0)
* cord         CORD v2 receipts, test split, as Parquet from the Hugging Face Hub (Clova AI, CC BY 4.0)
* tessdata     Tesseract's "fast" and "best" Hindi, English and script detection models (Apache 2.0)
* english      English sentences of `findnitai/english-to-hinglish` (Apache 2.0), for Latin lines
* mozhi_train  Mozhi train split (M2 trains on it)
* cord_train   CORD train and validation splits (M4 trains on them)

Every file's SHA256 goes into data/MANIFEST.json, so a result can name the exact data.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA  # noqa: E402
from doclens.text import normalize  # noqa: E402

MOZHI = "https://ilocr.iiit.ac.in/public/printed/phase-0/v0.5/hindi/akshara/{split}.zip"
TESSDATA = "https://github.com/tesseract-ocr/tessdata_{kind}/raw/main/{lang}.traineddata"
MANIFEST = DATA / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, source: str) -> None:
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man[str(path.relative_to(DATA))] = {"sha256": sha256(path), "bytes": path.stat().st_size, "source": source}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.relative_to(DATA)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "doclens"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r     {done / 1e6:7.1f} of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    record(dest, url)
    return dest


def get_mozhi(splits=("test", "val")) -> None:
    for split in splits:
        z = download(MOZHI.format(split=split), DATA / "mozhi" / "zips" / f"hindi_{split}.zip")
        out = DATA / "mozhi" / "hindi" / split
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        n = sum(1 for p in out.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        print(f"     mozhi {split}: {n} images")


def get_cord(splits=("test",)) -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download("naver-clova-ix/cord-v2", repo_type="dataset", local_dir=DATA / "cord",
                             allow_patterns=[f"data/{s}-*.parquet" for s in splits] + ["README.md"])
    files = sorted(p for s in splits for p in Path(path).rglob(f"{s}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {splits} in {path}")
    for p in files:
        record(p, "hf://datasets/naver-clova-ix/cord-v2")
        print(f"     cord {p.name}: {p.stat().st_size / 1e6:.1f} MB")


def get_tessdata() -> None:
    """Both model sets in data/, so every machine runs the same models whatever its package manager ships."""
    for kind in ("fast", "best"):
        for lang in ("hin", "eng", "osd"):
            download(TESSDATA.format(kind=kind, lang=lang), DATA / f"tessdata_{kind}" / f"{lang}.traineddata")


def get_english() -> None:
    from datasets import load_dataset
    dest = DATA / "text" / "english.txt"
    if dest.exists():
        print(f"have {dest.relative_to(DATA)}")
        return
    ds = load_dataset("findnitai/english-to-hinglish", split="train")
    seen, lines = set(), []
    for ex in ds:
        t = ex.get("translation") if isinstance(ex.get("translation"), dict) else ex
        s = normalize(str(t.get("en") or t.get("english") or t.get("English") or ""))
        if 10 <= len(s) <= 300 and s not in seen:
            seen.add(s)
            lines.append(s)
    if not lines:
        raise KeyError(f"no English field in {list(ds.features)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf8")
    record(dest, "hf://datasets/findnitai/english-to-hinglish (English side)")
    print(f"     english: {len(lines)} sentences")


STEPS = {"mozhi": get_mozhi, "cord": get_cord, "tessdata": get_tessdata, "english": get_english,
         "mozhi_train": lambda: get_mozhi(("train",)), "cord_train": lambda: get_cord(("train", "validation"))}
M1 = ["mozhi", "cord", "tessdata", "english"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(STEPS), default=M1)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for name in args.only:
        print(f"== {name}")
        STEPS[name]()
