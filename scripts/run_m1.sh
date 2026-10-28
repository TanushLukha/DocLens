#!/usr/bin/env bash
# M1 in one go: check the machine, fetch the data, build the synthetic sets, run the study.
#   bash scripts/run_m1.sh
# Run it with the environment active (source .venv/bin/activate).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname)" = "Darwin" ]; then
  # macOS removes DYLD variables when it starts bash, so set it again here:
  # Pillow finds FriBiDi (Devanagari shaping) in Homebrew's folder through it.
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:/usr/lib"
fi

python scripts/check_env.py
python scripts/get_data.py
python -m pytest -q
python scripts/make_synth.py
python scripts/eval_tesseract.py --quick
python scripts/eval_tesseract.py
echo "M1 done: docs/results/m1_tesseract.md"
