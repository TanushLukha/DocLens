#!/usr/bin/env bash
# M2 in one go: training data, preparation, tests, the benchmark, training, evaluation.
#   bash scripts/run_m2.sh                 (extra arguments go to train_crnn.py, e.g. --height 32)
# Run it with the environment active (source .venv/bin/activate), TinyGPT next to DocLens.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname)" = "Darwin" ]; then
  # macOS removes DYLD variables when it starts bash, so set it again here:
  # Pillow finds FriBiDi (Devanagari shaping) in Homebrew's folder through it.
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:/usr/lib"
fi

python scripts/check_env.py
python scripts/get_data.py --only mozhi_train cord_train
python scripts/prepare_m2.py --corpus ../TinyGPT/data/clean
python -m pytest -q
python scripts/bench_m2.py
python scripts/train_crnn.py --run main "$@"
python scripts/eval_crnn.py --run main
echo "M2 done: docs/results/m2_crnn.md"
