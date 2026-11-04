"""A models folder with tiny untrained models, shaped as scripts/export_models.py makes it.

    python scripts/tiny_models.py models          (CI builds the Docker image with these)

The tests and CI use it to check the service and its image end to end without the real models,
which are hundreds of megabytes and made on the Mac; what such a folder reads is nonsense, but every
step runs and every answer has its real shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .,")
LABELS = ["O", "B-total.total_price", "I-total.total_price"]


def tiny_crnn():
    import torch

    from doclens.alphabet import Alphabet
    from doclens.crnn import CRNN, CRNNConfig
    torch.manual_seed(0)
    cfg = CRNNConfig(n_classes=len(CHARS) + 1, height=32, channels=(8, 16, 32, 32), hidden=32)
    return CRNN(cfg).eval(), Alphabet(CHARS)


def write(folder: Path) -> Path:
    from doclens import kie
    from doclens.onnx_crnn import export
    folder = Path(folder)
    model, alphabet = tiny_crnn()
    export(model, alphabet, folder / "crnn.onnx")
    tok = kie.tiny_tokenizer(["TOTAL 9,100 first line of text a second line"] * 20, vocab=300)
    lilt = kie.build("lilt", LABELS, config=kie.tiny_config("lilt", len(tok)))
    lilt.save_pretrained(folder / "lilt")
    tok.save_pretrained(folder / "lilt")
    return folder


if __name__ == "__main__":
    print(f"tiny models in {write(Path(sys.argv[1] if len(sys.argv) > 1 else 'models'))}")
