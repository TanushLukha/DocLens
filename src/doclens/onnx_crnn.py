"""M2's CRNN as an ONNX graph, read with ONNX Runtime on the CPU (M6).

The graph takes the batch exactly as crnn.collate makes it (B x 1 x 48 x W, padded with paper, and
each image's width) and returns log probabilities (T x B x C) and columns per image; the batch and
the width may be any size. A JSON file beside the graph holds the alphabet and the input height, so
the graph and its alphabet travel together. OnnxRecognizer prepares, batches and decodes as
recognize.Recognizer does, so the two give the same text; only the network call differs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .alphabet import Alphabet
from .recognize import Recognizer


class _WithLogSoftmax(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, widths: torch.Tensor):
        logits, lengths = self.model(images, widths)
        return F.log_softmax(logits, dim=-1), lengths


def export(model, alphabet: Alphabet, path: str | Path) -> Path:
    """Writes the graph to `path` and its alphabet and sizes to the same name with .json. PyTorch's
    current exporter (torch.export, then onnxscript) with the batch and the width left free.
    Export one model per process: in PyTorch 2.14 a second export in the same process came out with
    the width fixed at the example's, so a graph whose batch or width is fixed is refused here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    h = model.cfg.height
    x = torch.ones(2, 1, h, 4 * h)
    widths = torch.tensor([4 * h, 2 * h])
    batch = torch.export.Dim("batch", min=1, max=4096)
    width = torch.export.Dim("width", min=16, max=16384)
    program = torch.onnx.export(_WithLogSoftmax(model).eval(), (x, widths), input_names=["images", "widths"],
                                output_names=["logprobs", "lengths"], dynamo=True, verbose=False,
                                dynamic_shapes={"images": {0: batch, 3: width}, "widths": {0: batch}})
    if program is None:
        raise RuntimeError("the ONNX exporter returned no program")
    program.save(str(path))
    import onnx
    dims = [[d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in onnx.load(str(path)).graph.input]
    if not (isinstance(dims[0][0], str) and isinstance(dims[0][3], str)):
        raise RuntimeError(f"the exported graph fixed its batch or width: inputs {dims}")
    meta = {"alphabet": "".join(alphabet.chars), "height": h, "max_width": model.cfg.max_width,
            "inputs": "images B x 1 x H x W float32 from minus 1 (ink) to plus 1 (paper), widths B int64"}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf8")
    return path


class OnnxRecognizer(Recognizer):
    """The CRNN through ONNX Runtime on the CPU: batches as large as they come, no blank fill."""

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.alphabet = Alphabet(meta["alphabet"])
        self.height, self.max_width = int(meta["height"]), int(meta["max_width"])
        self.device = torch.device("cpu")
        self.fill = False

    def _net(self, x: torch.Tensor, widths: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        logp, lengths = self.session.run(None, {"images": x.numpy(), "widths": widths.numpy()})
        return logp, lengths
