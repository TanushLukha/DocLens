"""Measures what M2's training costs on this machine, before any long run.

    python scripts/bench_m2.py

1. Checks that the GPU (Apple's MPS, or CUDA) runs every piece of a training step: the CRNN, CTC
   loss on the GPU (the CPU always works), and mixed precision in fp16 and bf16; then times the
   CRNN's LSTM (reversed within each length) against PyTorch's packed LSTM on lines of mixed length.
2. Times training steps at input heights 32 and 48 on batches as training builds them (as many
   images as fit 32,768 pixel columns): words, lines and the longest lines, with the GPU memory
   each one needs.
3. Times how fast one CPU worker draws training samples (rendering, damage, Mozhi crops).
4. Estimates samples per second for the training mix and hours per million samples.

Prints a report and saves nothing. Takes two to three minutes.
"""
import argparse
import contextlib
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import crnn  # noqa: E402
from doclens.alphabet import Alphabet  # noqa: E402
from doclens.crnn import CRNN, BiLSTM, CRNNConfig  # noqa: E402
from doclens.traindata import M2, Sources, Spec  # noqa: E402

ASPECT = {"words": 2, "lines": 9, "long lines": 22}     # width / height: a word, a typical line, the longest
CHARS = {"words": 4, "lines": 45, "long lines": 70}
SHARE = {"words": 0.62, "lines": 0.33, "long lines": 0.05}   # of training samples (Mozhi, CORD words, spans)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def fitting(kind: str, height: int, columns: int, max_batch: int = 256) -> int:
    """How many images of this kind a training batch holds (traindata.Sources.batches)."""
    w = -(-height * ASPECT[kind] // crnn.BUCKET) * crnn.BUCKET
    return max(1, min(max_batch, columns // w))


def gpu_gb(device) -> float:
    return torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else float("nan")


def batch(kind: str, height: int, n: int, classes: int, gen: torch.Generator):
    w = height * ASPECT[kind]
    x = torch.randn(n, 1, height, w, generator=gen)
    widths = torch.full((n,), w, dtype=torch.long)
    targets = torch.randint(1, classes, (n * CHARS[kind],), generator=gen)
    lengths = torch.full((n,), CHARS[kind], dtype=torch.long)
    return x, widths, targets, lengths


def step_fn(model, opt, device, b, amp=None, ctc_on_cpu=True):
    x, widths, targets, lengths = b
    with torch.autocast(device.type, dtype=amp) if amp is not None else contextlib.nullcontext():
        logits, cols = model(x.to(device), widths.to(device))
    loss = crnn.ctc_loss(logits, cols, targets.to(device), lengths.to(device), on_cpu=ctc_on_cpu)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def check(name, fn):
    try:
        out = fn()
        print(f"ok   {name}" + (f": {out}" if out is not None else ""))
        return True
    except Exception as e:                                  # report and go on: this is a probe
        print(f"NO   {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def timed(model, opt, device, b, steps=10, warmup=3, **kw):
    for _ in range(warmup):
        step_fn(model, opt, device, b, **kw)
    sync(device)
    t = time.perf_counter()
    for _ in range(steps):
        step_fn(model, opt, device, b, **kw)
    sync(device)
    return (time.perf_counter() - t) / steps


def lstm_compare(device, steps=5):
    """Forward and backward through two LSTM layers each way, 64 sequences of 100 to 300 columns."""
    torch.manual_seed(0)
    cols, n, feats, hidden = 300, 64, 256, 256
    lengths = torch.randint(cols // 3, cols + 1, (n,))
    lengths[0] = cols
    x = torch.randn(cols, n, feats, device=device)
    ours = BiLSTM(feats, hidden, 2, 0.0).to(device)
    ref = nn.LSTM(feats, hidden, num_layers=2, bidirectional=True).to(device)

    def run_ours():
        ours(x, lengths).sum().backward()

    def run_packed():
        seq = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        out, _ = ref(seq)
        nn.utils.rnn.pad_packed_sequence(out, total_length=cols)[0].sum().backward()

    def clock(fn):
        for _ in range(2):
            fn()
        sync(device)
        t = time.perf_counter()
        for _ in range(steps):
            fn()
        sync(device)
        return (time.perf_counter() - t) / steps

    try:
        a, b = clock(run_ours), clock(run_packed)
        print(f"LSTM, 64 lines of 100 to 300 columns, forward and backward: reversed within length {1000 * a:.0f} ms,"
              f" PyTorch packed {1000 * b:.0f} ms ({b / a:.1f} times slower)")
    except Exception as e:                                  # report and go on: this is a probe
        print(f"NO   LSTM comparison: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--columns", type=int, default=32768, help="pixel columns per batch, as in training")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    device = crnn.pick_device(args.device)
    print(f"torch {torch.__version__}, Python {platform.python_version()}, {platform.platform()}, device {device}")
    abc_file = M2 / "alphabet.json"
    classes = len(Alphabet.load(abc_file)) if abc_file.exists() else 260
    gen = torch.Generator().manual_seed(0)

    print("\n1. what runs on the device")
    model = CRNN(CRNNConfig(n_classes=classes, height=48)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    small = batch("lines", 48, 8, classes, gen)
    check("CRNN step, CTC on the CPU", lambda: f"loss {step_fn(model, opt, device, small):.2f}")
    ctc_dev = check("CTC loss on the device", lambda: f"loss {step_fn(model, opt, device, small, ctc_on_cpu=False):.2f}")
    amps = {"off": None}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        if device.type == "cpu" and name == "fp16":
            continue
        ok = check(f"mixed precision {name}", lambda dt=dt: f"loss {step_fn(model, opt, device, small, amp=dt):.2f}")
        if ok:
            amps[name] = dt

    lstm_compare(device)

    print(f"\n2. training steps on batches of {args.columns:,} pixel columns, full precision"
          " (images per batch, ms per step, samples per second, GPU memory)")
    times: dict = {}
    for height in (32, 48):
        torch.manual_seed(0)
        model = CRNN(CRNNConfig(n_classes=classes, height=height)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for kind in ASPECT:
            n = fitting(kind, height, args.columns)
            if device.type == "mps":
                torch.mps.empty_cache()
            try:
                s = timed(model, opt, device, batch(kind, height, n, classes, gen))
                times[(height, kind)] = s / n
                print(f"height {height}  {kind:>10}: {n:3d} images  {1000 * s:5.0f} ms  {n / s:6.0f}/s"
                      f"  {gpu_gb(device):5.1f} GB")
            except Exception as e:
                print(f"height {height}  {kind:>10}: failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    if ctc_dev:
        model = CRNN(CRNNConfig(n_classes=classes, height=48)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        n = fitting("lines", 48, args.columns)
        on_dev = timed(model, opt, device, batch("lines", 48, n, classes, gen), ctc_on_cpu=False)
        print(f"lines at 48 with CTC on the device: {1000 * on_dev:.0f} ms per step"
              f" (CPU CTC: {1000 * n * times.get((48, 'lines'), float('nan')):.0f} ms)")

    print("\n3. one data worker (samples per second, rendering and damage on the CPU)")
    rates = {}
    if abc_file.exists():
        abc = Alphabet.load(abc_file)
        for source in ("lines", "mozhi"):
            spec = Spec(alphabet="".join(abc.chars), height=48, max_width=48 * 24, weights={source: 1.0})
            try:
                src = Sources(spec)
                rng = np.random.default_rng(0)
                t = time.perf_counter()
                for _ in range(200):
                    src.sample(rng)
                rates[source] = 200 / (time.perf_counter() - t)
                print(f"{source:>6}: {rates[source]:6.0f} samples/s per worker")
            except Exception as e:
                print(f"{source:>6}: skipped ({type(e).__name__}: {e})")
    else:
        print("no data/m2/alphabet.json yet: run scripts/prepare_m2.py first")

    print("\n4. estimate for the training mix (words 62%, lines 33%, the longest lines 5% of samples)")
    for height in (32, 48):
        if all((height, k) in times for k in ASPECT):
            rate = 1 / sum(SHARE[k] * times[(height, k)] for k in ASPECT)
            print(f"height {height}: about {rate:5.0f} samples/s on the device, {1e6 / rate / 3600:4.1f} h per million samples")
    if rates:
        print(f"one worker draws about {rates.get('lines', float('nan')):.0f} line samples/s; "
              "use enough workers to stay above the device's rate")


if __name__ == "__main__":
    main()
