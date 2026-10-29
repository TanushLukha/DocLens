"""Checks everything M1 needs before any long run: Python packages, Devanagari shaping, Tesseract.

    python scripts/check_env.py
"""
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import DATA, fonts, tesseract  # noqa: E402

ok = True


def report(name: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"{'ok ' if good else 'NO '} {name}{': ' + detail if detail else ''}")


print(f"Python {platform.python_version()} on {platform.platform()}")
FOR_DOWNLOADS = ("pyarrow", "huggingface_hub", "datasets")
for mod in ("numpy", "cv2", "PIL", "regex", "fontTools") + FOR_DOWNLOADS:
    try:
        m = importlib.import_module(mod)
        report(mod, True, getattr(m, "__version__", ""))
    except ImportError:
        hint = " (needed by get_data.py: pip install -e \".[data]\")" if mod in FOR_DOWNLOADS else ""
        report(mod, mod in FOR_DOWNLOADS, "missing" + hint)

from PIL import features  # noqa: E402

report("Pillow can shape Devanagari (libraqm)", features.check("raqm"), str(features.version("raqm")))
if not features.check("raqm"):
    print("    Pillow's own libraqm loads FriBiDi when the program starts. On a Mac:\n"
          "      brew install fribidi\n"
          "      echo 'export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib' >> ~/.zshrc\n"
          "      source ~/.zshrc\n"
          "    (Pillow looks for libfribidi.dylib by name and in /usr/local/lib; Homebrew on Apple silicon\n"
          "    keeps it in /opt/homebrew/lib, which is not searched unless this variable says so.)")
else:
    bad = [f.name for f in fonts.load(script="deva") if not fonts.shaping_works(f)]
    report("every Devanagari font shapes conjuncts", not bad, ", ".join(bad))
report("fonts", True, f"{len(fonts.load())} files, {len(fonts.load(split='test'))} held out for testing")

try:
    report("tesseract", True, tesseract.version())
    for kind in ("fast", "best"):
        folder = DATA / f"tessdata_{kind}"
        if folder.exists():
            langs = tesseract.languages(str(folder))
            report(f"tesseract {kind} models: hin and eng", {"hin", "eng"} <= set(langs), " ".join(langs))
        else:
            print(f"... tesseract {kind} models not downloaded yet (scripts/get_data.py)")
except RuntimeError as e:
    report("tesseract", False, str(e))

try:
    import torch
    report("torch", True, f"{torch.__version__}, Apple GPU (MPS): {torch.backends.mps.is_available()}")
except ImportError:
    print("... torch not installed (needed from M2)")

print("\nall good" if ok else "\nfix the lines marked NO first")
sys.exit(0 if ok else 1)
