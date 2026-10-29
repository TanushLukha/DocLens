"""M2: trains the CRNN recognizer with CTC loss.

    python scripts/train_crnn.py --run main                 the main model (all three sources)
    python scripts/train_crnn.py --run synthetic --synthetic-only
    python scripts/train_crnn.py --run main --resume        continue an interrupted run
    python scripts/train_crnn.py --run smoke --smoke        a two minute check of the whole loop

Writes runs/m2/<run>/: config.json, alphabet.json, last.pt (every evaluation), best.pt (lowest
mean validation CER), metrics.jsonl (one line per log and evaluation), results.json at the end.
With MLflow installed (pip install -e ".[track]"), every run is also recorded in runs/mlflow.db:
    mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
"""
import argparse
import contextlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, crnn  # noqa: E402
from doclens.alphabet import Alphabet  # noqa: E402
from doclens.crnn import CRNN, CRNNConfig  # noqa: E402
from doclens.metrics import Scores  # noqa: E402
from doclens.recognize import Recognizer  # noqa: E402
from doclens.traindata import M2, Batches, Spec, val_sets  # noqa: E402

RUNS = REPO / "runs" / "m2"
MIX = {"lines": 0.5, "mozhi": 0.3, "cord": 0.2}       # share of training samples from each source


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--channels", type=int, nargs=4, default=[32, 64, 128, 256])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--columns", type=int, default=32768, help="pixel columns per batch: images times width")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    ap.add_argument("--ctc-on-device", action="store_true", help="CTC on the GPU instead of the CPU")
    ap.add_argument("--sources", nargs="+", choices=list(MIX), default=list(MIX), help="training sources to mix")
    ap.add_argument("--synthetic-only", action="store_true", help="the same as --sources lines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="a few steps on a small batch, to check the loop")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.columns, args.max_batch, args.eval_every, args.log_every, args.warmup = 60, 4096, 32, 30, 10, 10
    return args


def lr_at(step: int, args) -> float:
    """Linear warm up, then a cosine down to min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def memory(device: torch.device) -> dict:
    """GPU memory held by PyTorch, and on a Mac the swap in use: a run that swaps slows to a crawl."""
    out = {}
    if device.type == "mps":
        out["gpu_gb"] = torch.mps.driver_allocated_memory() / 1e9
    elif device.type == "cuda":
        out["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if sys.platform == "darwin":
        try:
            text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
            used = re.search(r"used = ([\d.]+)M", text)
            if used:
                out["swap_gb"] = float(used.group(1)) / 1024
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def evaluate(model, alphabet, device, vals: dict, limit: int | None = None) -> dict:
    model.eval()
    rec = Recognizer(model, alphabet, device)
    out = {}
    for name, items in vals.items():
        items = items[:limit] if limit else items
        hyps = rec.read([it.image() for it in items])
        sc = Scores()
        for it, h in zip(items, hyps, strict=True):
            sc.add(it.text, h)
        out[name] = sc.summary()
        out[name]["examples"] = [[it.text, h] for it, h in list(zip(items, hyps, strict=True))[:3]]
    model.train()
    return out


class Tracker:
    """MLflow when it is installed and wanted; otherwise nothing."""

    def __init__(self, args, out: Path, params: dict):
        self.ml = None
        if args.no_mlflow:
            return
        try:
            import mlflow
        except ImportError:
            print("MLflow not installed: metrics go to metrics.jsonl only (pip install -e \".[track]\" for MLflow)")
            return
        mlflow.set_tracking_uri(f"sqlite:///{REPO / 'runs' / 'mlflow.db'}")
        name = "doclens_m2"
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(REPO / "runs" / "mlartifacts").as_uri())
        mlflow.set_experiment(name)
        id_file = out / "mlflow_run_id"
        run_id = id_file.read_text().strip() if args.resume and id_file.exists() else None
        run = mlflow.start_run(run_id=run_id, run_name=args.run)
        id_file.write_text(run.info.run_id)
        if run_id is None:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        self.ml = mlflow

    def metrics(self, values: dict, step: int) -> None:
        if self.ml:
            self.ml.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    def finish(self, out: Path) -> None:
        if self.ml:
            for f in ("config.json", "alphabet.json", "metrics.jsonl", "results.json"):
                if (out / f).exists():
                    self.ml.log_artifact(str(out / f))
            self.ml.end_run()


def main():
    args = parse()
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = crnn.pick_device(args.device)
    alphabet = Alphabet.load(M2 / "alphabet.json")
    cfg = CRNNConfig(n_classes=len(alphabet), height=args.height, channels=tuple(args.channels), hidden=args.hidden)
    model = CRNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type) if args.amp == "fp16" else None
    sources = ["lines"] if args.synthetic_only else args.sources
    weights = {k: (v if k in sources else 0.0) for k, v in MIX.items()}
    if weights["cord"] and not (M2 / "cord_train" / "labels.tsv").exists():
        if not args.smoke:
            sys.exit("no CORD training crops: python scripts/get_data.py --only cord_train, then python scripts/prepare_m2.py"
                     " (or leave CORD out: --sources lines mozhi)")
        weights["cord"] = 0.0
    spec = Spec(alphabet="".join(alphabet.chars), height=cfg.height, max_width=cfg.max_width, columns=args.columns,
                max_batch=args.max_batch,
                weights=weights, seed=args.seed)

    start, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start, best = int(ckpt["step"]), float(ckpt["best"])
        torch.set_rng_state(ckpt["rng"])
        print(f"resumed {args.run} at step {start}, best mean validation CER {100 * best:.2f}%")

    params = {**vars(args), "params": crnn.params(model), "classes": len(alphabet), "device": str(device),
              "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
              "weights": weights, "max_width": cfg.max_width}
    (out / "config.json").write_text(json.dumps({"args": params, "model": vars(cfg) | {"channels": list(cfg.channels)},
                                                 "data": vars(spec) | {"alphabet": len(alphabet.chars)}},
                                                indent=1, ensure_ascii=False, default=str))
    alphabet.save(out / "alphabet.json")
    tracker = Tracker(args, out, params)
    vals = val_sets()
    limit = 50 if args.smoke else None
    print(f"{args.run}: {crnn.params(model):,} parameters, {len(alphabet)} classes, height {cfg.height},"
          f" device {device}, amp {args.amp}; validation: {({k: len(v) for k, v in vals.items()})}")

    workers = 0 if args.smoke else args.workers
    # a pool gives about 8 batches, so a resumed run starts near the pools it has not used yet
    loader = DataLoader(Batches(spec, start_pool=start // 8), batch_size=None, num_workers=workers,
                        prefetch_factor=4 if workers else None, persistent_workers=False)
    stream = iter(loader)
    log = open(out / "metrics.jsonl", "a" if args.resume else "w", encoding="utf8")    # a fresh run starts a fresh log
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)
    window = {"loss": 0.0, "n": 0, "samples": 0, "wait": 0.0, "t": time.time(), "sources": {}}
    model.train()
    for step in range(start, args.steps):
        t = time.time()
        batch = next(stream)
        window["wait"] += time.time() - t
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, widths = batch["x"].to(device), batch["widths"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext():
            logits, lengths = model(x, widths)
        per = args.columns / 512            # as if 64 lines of 512 pixels: every image weighs the same
        loss = crnn.ctc_loss(logits, lengths, batch["targets"], batch["target_lengths"], on_cpu=not args.ctc_on_device,
                             per=per)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        window["loss"] += float(loss.detach()) * per / len(batch["widths"])     # logged: the mean per image
        window["n"] += 1
        window["samples"] += len(batch["widths"])
        for k, v in batch["sources"].items():
            window["sources"][k] = window["sources"].get(k, 0) + v

        done = step + 1
        if done % args.log_every == 0:
            secs = time.time() - window["t"]
            rec = {"step": done, "loss": window["loss"] / window["n"], "lr": lr, "grad_norm": float(gnorm),
                   "samples_per_s": window["samples"] / secs, "data_wait": window["wait"] / secs,
                   "batch": window["samples"] / window["n"], **memory(device), "sources": window["sources"]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            tracker.metrics({k: rec[k] for k in ("loss", "lr", "grad_norm", "samples_per_s", "data_wait")}, done)
            eta = (args.steps - done) * secs / window["n"] / 60
            mem = "".join(f"  {k[:-3]} {rec[k]:.1f} GB" for k in ("gpu_gb", "swap_gb") if k in rec)
            print(f"step {done:6d}  loss {rec['loss']:.3f}  lr {lr:.2e}  {rec['samples_per_s']:5.0f} samples/s"
                  f"  batch {rec['batch']:4.0f}{mem}  data wait {100 * rec['data_wait']:3.0f}%  eta {eta:5.0f} min",
                  flush=True)
            window = {"loss": 0.0, "n": 0, "samples": 0, "wait": 0.0, "t": time.time(), "sources": {}}

        if done % args.eval_every == 0 or done == args.steps:
            res = evaluate(model, alphabet, device, vals, limit)
            mean = sum(r["cer"] for r in res.values()) / max(len(res), 1) if res else float("inf")
            improved = mean < best
            best = min(best, mean)
            extra = {"optimizer": opt.state_dict(), "step": done, "best": best, "rng": torch.get_rng_state(),
                     "val": {k: {m: v[m] for m in ("cer", "wer", "exact")} for k, v in res.items()}}
            if scaler is not None:
                extra["scaler"] = scaler.state_dict()
            crnn.save(out / "last.pt", model, alphabet, **extra)
            if improved:
                crnn.save(out / "best.pt", model, alphabet, step=done, val=extra["val"], best=best)
            log.write(json.dumps({"step": done, "val": res, "mean_cer": mean}, ensure_ascii=False) + "\n")
            log.flush()
            tracker.metrics({f"val_{k}_cer": v["cer"] for k, v in res.items()} | {"val_mean_cer": mean}, done)
            print("eval  " + "  ".join(f"{k} {100 * v['cer']:.2f}%" for k, v in res.items())
                  + f"  mean {100 * mean:.2f}%" + ("  (best, saved)" if improved else ""), flush=True)

    log.close()
    final = json.loads((out / "metrics.jsonl").read_text(encoding="utf8").splitlines()[-1])
    (out / "results.json").write_text(json.dumps({"run": args.run, "steps": args.steps, "best_mean_cer": best,
                                                  "last": final}, indent=1, ensure_ascii=False))
    tracker.finish(out)
    print(f"done: best mean validation CER {100 * best:.2f}%, checkpoints in {out}")


if __name__ == "__main__":
    main()
