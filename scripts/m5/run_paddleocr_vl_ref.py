"""M5: PaddleOCR VL 1.6's model through its reference implementation, on a part of the crops.

    python scripts/m5/run_paddleocr_vl_ref.py           (the main environment: PyTorch and Transformers)

A check beside the comparison, not a seventh system (decision 48 in docs/PLAN.md). On the Mac the
PaddleOCR VL pipeline serves its model through MLX VLM, as its Apple silicon guide sets it up, and
scripts/m5/probe.py found that path reading Devanagari worse than Hugging Face Transformers, the
model's reference implementation, on the same crops. So the first 200 Mozhi words, the first 200
synthetic lines and the first 100 CORD lines are read again here: the model in Transformers in
float32 on the CPU (the probe's most faithful setting), each crop sized as the pipeline sizes a
text block (112,896 to 1,003,520 pixels) and read with its OCR prompt, as the pipeline does with no
layout step. scripts/m5_score.py sets these readings beside the pipeline's on the same crops.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
SUBSET = {"mozhi": 200, "synth": 200, "cord_line": 100}
MIN_PIXELS, MAX_PIXELS = 112896, 1003520


def main():
    a = common.args(__doc__)
    if a.task != "crops":
        print("the reference check reads crops only")
        return
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sets = [s for s in SUBSET if not a.sets or s in a.sets]
    rows = [r for s in sets for r in common.pick(common.load("crops"), [s], min(SUBSET[s], a.limit or SUBSET[s]))]
    tag = f"transformers {transformers.__version__} {MODEL} float32 cpu"
    results = common.Results("paddleocr_vl_ref", "crops", tag)
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    load_s = time.perf_counter() - t0
    size = {"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}}

    def read(part):
        out = []
        for r in part:
            msgs = [{"role": "user", "content": [{"type": "image", "image": common.image(r, "RGB")},
                                                 {"type": "text", "text": "OCR:"}]}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt", processor_kwargs={"images_kwargs": size})
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            out.append({"text": " ".join(proc.decode(ids[0][inputs["input_ids"].shape[-1]:],
                                                     skip_special_tokens=True).split())})
        return out

    n = 0
    for name, rs in common.by_set(rows).items():
        print(f"paddleocr_vl_ref crops {name}: {len(rs)} images", flush=True)
        n += common.run(results, rs, read, batch=1, show=a.show, every=50)
    info = {"tag": tag, "read": n, "load_seconds": load_s, "memory": common.memory(), "device": "cpu, float32",
            "subset": SUBSET, "versions": common.versions(["torch", "transformers"])}
    print(f"paddleocr_vl_ref crops: {n} read; meta in {common.record('paddleocr_vl_ref', 'crops', info)}")


if __name__ == "__main__":
    main()
