"""M6: the models the service loads, in one folder.

    python scripts/export_models.py              models/: crnn.onnx and crnn.json, lilt/, MANIFEST.json

The CRNN (runs/m2/main/best.pt) becomes an ONNX graph (doclens.onnx_crnn), checked against PyTorch
on every eighth M5 crop: both must write the same text, and their log probabilities must agree to
within 0.001. The same crops are read by an int8 copy of the graph too (ONNX Runtime's dynamic
quantisation), to measure what smaller weights would cost; the copy is not kept. The field
extractor is LiLT trained on CORD's true words (M4): of its three seeds,
the one with the best validation field F1 (the test receipts are never used to choose), saved
with its labels and RoBERTa's tokenizer, so the service downloads nothing. MANIFEST.json gives
every file's SHA256 and where it came from.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "m5"))

from doclens import REPO  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crnn(src: Path, out: Path) -> dict:
    import common

    from doclens import crnn as crnn_mod
    from doclens.ctc import greedy
    from doclens.onnx_crnn import OnnxRecognizer, export
    from doclens.recognize import Recognizer
    model, alphabet, _ = crnn_mod.load(src, "cpu")
    path = export(model, alphabet, out / "crnn.onnx")
    rows = common.load("crops")[::8]
    images = [common.array(r) for r in rows]
    torch_rec, onnx_rec = Recognizer(model, alphabet, "cpu"), OnnxRecognizer(path)
    t0 = time.perf_counter()
    a = dict(torch_rec.logprobs(images, 64))
    t1 = time.perf_counter()
    b = dict(onnx_rec.logprobs(images, 64))
    t2 = time.perf_counter()
    diff = max(float(np.abs(a[i] - b[i]).max()) for i in a)
    same = sum(alphabet.decode(greedy(a[i])) == onnx_rec.alphabet.decode(greedy(b[i])) for i in a)
    print(f"CRNN: {path} ({path.stat().st_size / 2**20:.1f} MB); {same} of {len(a)} crops read the same, largest"
          f" log probability difference {diff:.6f}; CPU {t1 - t0:.1f} s with PyTorch, {t2 - t1:.1f} s with ONNX Runtime")
    if same != len(a) or diff > 0.001:
        raise SystemExit("the ONNX graph does not read like the PyTorch model: not exported")
    info = {"source": str(src.resolve().relative_to(REPO)), "source_sha256": sha(src), "crops_checked": len(a),
            "same_text": same, "max_logprob_diff": diff, "seconds_torch_cpu": t1 - t0, "seconds_onnx_cpu": t2 - t1,
            "cer": cer(rows, onnx_rec.read(images)), "mb": path.stat().st_size / 2**20}
    info["int8_trial"] = int8_trial(path, rows, images)
    t = info["int8_trial"]
    print(f"   int8 copy: {t['mb']:.1f} MB against {info['mb']:.1f}, {t['seconds']:.1f} s against {t2 - t1:.1f} s,"
          f" CER {100 * t['cer']:.2f}% against {100 * info['cer']:.2f}%: not kept")
    return info


def cer(rows: list[dict], texts: list[str]) -> float:
    from doclens.metrics import Scores
    sc = Scores()
    for r, t in zip(rows, texts, strict=True):
        sc.add(r["text"], t)
    return sc.summary()["cer"]


def int8_trial(path: Path, rows: list[dict], images: list) -> dict:
    import shutil
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    from doclens.onnx_crnn import OnnxRecognizer
    with tempfile.TemporaryDirectory() as tmp:
        q = Path(tmp) / "crnn.onnx"
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
        shutil.copy(path.with_suffix(".json"), q.with_suffix(".json"))
        rec = OnnxRecognizer(q)
        t0 = time.perf_counter()
        texts = rec.read(images, 64)
        return {"seconds": time.perf_counter() - t0, "cer": cer(rows, texts), "mb": q.stat().st_size / 2**20}


def lilt(runs: Path, out: Path) -> dict:
    import torch
    from transformers import AutoConfig

    from doclens import kie
    best = None
    for p in sorted(runs.glob("lilt_word_true_s*/best.pt")):
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        if best is None or ckpt["best"]["val_f1"] > best[1]["best"]["val_f1"]:
            best = (p, ckpt)
    if best is None:
        raise SystemExit(f"no LiLT run in {runs}: run scripts/run_m4.sh first")
    p, ckpt = best
    model = kie.build("lilt", ckpt["labels"], config=AutoConfig.from_pretrained(kie.MODELS["lilt"]))
    model.load_state_dict(ckpt["model"])
    folder = out / "lilt"
    model.save_pretrained(folder)
    kie.tokenizer(ckpt["tokenizer"]).save_pretrained(folder)
    print(f"LiLT: {p.parent.name} (validation field F1 {100 * ckpt['best']['val_f1']:.1f}%, step {ckpt['best']['step']})"
          f" -> {folder}")
    return {"run": p.parent.name, "source_sha256": sha(p), "val_f1": ckpt["best"]["val_f1"], "step": ckpt["best"]["step"],
            "labels": len(ckpt["labels"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crnn", default=str(REPO / "runs" / "m2" / "main" / "best.pt"))
    ap.add_argument("--runs", default=str(REPO / "runs" / "m4"), help="M4's training runs")
    ap.add_argument("--out", default=str(REPO / "models"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"crnn": crnn(Path(a.crnn), out), "lilt": lilt(Path(a.runs), out)}
    manifest["files"] = {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*"))
                         if f.is_file() and f.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print(f"wrote {out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
