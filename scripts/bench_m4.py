"""M4: checks before the long runs, on the Mac that will train.

    python scripts/bench_m4.py

1. The three models read the same vocabulary: LiLT's and LayoutLMv3's own tokenizers against
   RoBERTa's, which encodes the words for all three (decision 39 in docs/PLAN.md).
2. How long CORD's training receipts are in tokens, and so how many windows fall in each of the
   three fixed lengths (128, 256, 512).
3. How fast and how big each model trains: a few optimiser steps per length at the training batch
   size, with the GPU memory they take, and what a run of the planned length would cost.
Results to runs/m4/bench.json.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO, kie  # noqa: E402
from doclens.crnn import pick_device  # noqa: E402
from doclens.fields import all_labels, cord_receipts  # noqa: E402


def gb(x: float) -> float:
    return round(x / 2**30, 2)


def memory(device) -> dict:
    if device.type == "mps":
        return {"allocated_gb": gb(torch.mps.current_allocated_memory()), "driver_gb": gb(torch.mps.driver_allocated_memory())}
    if device.type == "cuda":
        return {"allocated_gb": gb(torch.cuda.max_memory_allocated())}
    return {}


def sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(kie.MODELS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6, help="timed steps per length, after 2 warm up steps")
    ap.add_argument("--planned-steps", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    out: dict = {"device": str(device), "torch": torch.__version__, "batch": args.batch}

    tok = kie.tokenizer()
    from transformers import AutoTokenizer
    vocab = tok.get_vocab()
    out["vocab"] = {}
    for kind in ("lilt", "layoutlmv3"):
        try:
            own = AutoTokenizer.from_pretrained(kie.MODELS[kind]).get_vocab()
        except (OSError, ValueError) as e:                 # a checkpoint without tokenizer files of its own
            out["vocab"][kind] = {"error": str(e)[:300]}
            print(f"{kind}: no tokenizer of its own ({str(e)[:120]})", flush=True)
            continue
        out["vocab"][kind] = {"size": len(own), "same_as_roberta": own == vocab}
        print(f"{kind}'s own vocabulary: {len(own)} tokens, the same as RoBERTa's: {own == vocab}", flush=True)

    labels = all_labels()
    train = cord_receipts("train", image="small")
    ids = {t: i for i, t in enumerate(labels)}
    ex = [e for r in train for e in kie.encode(tok, r, ids)]
    real = [int(e.mask.sum()) for e in ex]
    lengths = Counter(len(e.ids) for e in ex)
    out["receipts"] = len(train)
    out["windows"] = {str(k): v for k, v in sorted(lengths.items())}
    out["tokens"] = {"median": float(np.median(real)), "p95": float(np.percentile(real, 95)), "max": int(max(real))}
    print(f"{len(train)} training receipts, {len(ex)} windows; windows per length {dict(sorted(lengths.items()))};"
          f" tokens per window median {np.median(real):.0f}, 95th percentile {np.percentile(real, 95):.0f}, max {max(real)}",
          flush=True)
    share = {k: (v // args.batch) for k, v in lengths.items()}
    total = sum(share.values())

    out["models"] = {}
    for kind in args.models:
        model = kie.build(kind, labels).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        row: dict = {"parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "seconds_per_step": {}}
        for length in sorted(lengths):
            group = [e for e in ex if len(e.ids) == length]
            batch = (group * args.batch)[:args.batch]
            t = kie.tensors(batch, kind, device)
            for i in range(2 + args.steps):
                if i == 2:
                    sync(device)
                    t0 = time.perf_counter()
                loss = model(**t).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            sync(device)
            row["seconds_per_step"][str(length)] = round((time.perf_counter() - t0) / args.steps, 3)
        row["memory"] = memory(device)
        per_step = sum(row["seconds_per_step"][str(k)] * n for k, n in share.items()) / max(total, 1)
        row["mean_seconds_per_step"] = round(per_step, 3)
        row["planned_run_minutes"] = round(per_step * args.planned_steps / 60, 1)
        out["models"][kind] = row
        print(f"{kind}: {row['parameters_m']} M parameters; seconds per step by length {row['seconds_per_step']};"
              f" memory {row['memory']}; about {row['planned_run_minutes']} minutes for {args.planned_steps} steps"
              " (validation not included)", flush=True)
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()
    path = REPO / "runs" / "m4" / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
