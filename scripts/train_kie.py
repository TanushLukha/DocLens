"""M4: trains one field extractor on CORD's training receipts.

    python scripts/train_kie.py --model lilt --seed 0
    python scripts/train_kie.py --model roberta --tiny --steps 40      a tiny model, nothing downloaded: checks the code

The model reads the true words of the 800 training receipts (or, when asked, their OCR words,
tagged from the true words under them) and is scored on the 100 validation receipts every 100
steps by field F1; the weights with the best score are kept (decision 40 in docs/PLAN.md). Output
in runs/m4/<run>/: best.pt (weights, labels, settings), history.json. With MLflow installed, the
run is also recorded in runs/mlflow.db (experiment doclens_m4).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, kie  # noqa: E402
from doclens.crnn import pick_device  # noqa: E402
from doclens.fields import all_labels, cord_receipts, ocr_receipts, transfer  # noqa: E402

RUNS = REPO / "runs" / "m4"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(kie.MODELS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=None,
                    help="parts a batch goes through the model in (default: 4 for LayoutLMv3, whose whole batches "
                         "ran out of GPU memory on the Mac; the whole batch for the others)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--boxes", choices=["word", "segment"], default="word",
                    help="segment: every word gets its field's box, the papers' setting (a check, decision 40)")
    ap.add_argument("--train-on", choices=["true", "ocr"], default="true")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--label-splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="use only this many receipts of each split")
    ap.add_argument("--tiny", action="store_true", help="a tiny model and tokenizer trained here: checks the code")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run", default=None, help="run name (default: model, boxes, input and seed)")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    a.run = a.run or (f"{a.model}_{a.boxes}_{a.train_on}_s{a.seed}" + ("_tiny" if a.tiny else ""))
    a.micro = a.micro or (4 if a.model == "layoutlmv3" and a.batch % 4 == 0 else a.batch)
    return a


class Tracker:
    """MLflow when it is installed and wanted; otherwise nothing."""

    def __init__(self, args, params: dict):
        self.ml = None
        if args.no_mlflow or args.tiny:
            return
        try:
            import mlflow
        except ImportError:
            print('MLflow not installed: metrics go to history.json only (pip install -e ".[track]" for MLflow)')
            return
        mlflow.set_tracking_uri(f"sqlite:///{REPO / 'runs' / 'mlflow.db'}")
        name = "doclens_m4"
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(REPO / "runs" / "mlartifacts").as_uri())
        mlflow.set_experiment(name)
        mlflow.start_run(run_name=args.run)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        self.ml = mlflow

    def log(self, row: dict) -> None:
        gpu = f", GPU {row['gpu_gb']} GB" if "gpu_gb" in row else ""
        print(f"   step {row['step']:5d}  loss {row['loss']:.4f}  validation field F1 {100 * row['val_f1']:.2f}"
              f"  ({row['seconds'] / 60:.1f} min{gpu})", flush=True)
        if self.ml:
            self.ml.log_metrics({k: float(v) for k, v in row.items() if k != "step"}, step=row["step"])

    def finish(self, out: Path) -> None:
        if self.ml:
            self.ml.log_artifact(str(out / "history.json"))
            self.ml.end_run()


def load(args, kind: str):
    image = "small" if kind == "layoutlmv3" else "none"
    lim = slice(None, args.limit)
    train = cord_receipts(args.train_split, image=image)[lim]
    val = cord_receipts(args.val_split, image=image)[lim]
    if args.train_on == "ocr":
        train = [transfer(o, t) for o, t in zip(ocr_receipts(args.train_split, train), train, strict=True)]
    return train, val


def main():
    args = parse()
    kind = args.model
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    labels = all_labels(args.label_splits)
    train, val = load(args, kind)
    if args.tiny:
        tok = kie.tiny_tokenizer([" ".join(r.words) for r in train], vocab=2000)
        model = kie.build(kind, labels, config=kie.tiny_config(kind, len(tok)))
    else:
        tok = kie.tokenizer()
        model = kie.build(kind, labels)
    ids = {t: i for i, t in enumerate(labels)}

    def windows(receipts):
        return [e for r in receipts for e in kie.encode(tok, r, ids, kie.segment_boxes(r) if args.boxes == "segment" else None)]

    tex, vex = windows(train), windows(val)
    print(f"{args.run}: {len(train)} training receipts ({len(tex)} windows), {len(val)} validation receipts;"
          f" {len(labels)} labels; {args.steps} steps of {args.batch} (in parts of {args.micro})"
          f" = {kie.epochs(tex, args.batch, args.steps):.1f} epochs;"
          f" {device}; loaded in {time.time() - t0:.0f} s", flush=True)
    params = {**vars(args), "labels": len(labels), "train_windows": len(tex), "device": str(device),
              "torch": torch.__version__, "checkpoint": kie.MODELS[kind] if not args.tiny else "tiny"}
    tracker = Tracker(args, params)
    state, history = kie.train(model, kind, tex, (val, vex), labels, device, args.steps, args.batch, args.lr, args.warmup,
                               args.eval_every, args.seed, tracker.log, args.micro)
    best = max(history, key=lambda h: h["val_f1"])
    torch.save({"model": state, "labels": labels, "args": vars(args), "kind": kind, "best": best,
                "tokenizer": None if args.tiny else kie.TOKENIZER}, out / "best.pt")
    if args.tiny:
        tok.save_pretrained(out / "tokenizer")
    (out / "history.json").write_text(json.dumps({"params": params, "history": history, "best": best}, indent=1),
                                      encoding="utf8")
    tracker.finish(out)
    print(f"{args.run}: best validation field F1 {100 * best['val_f1']:.2f} at step {best['step']};"
          f" {(time.time() - t0) / 60:.1f} min -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
