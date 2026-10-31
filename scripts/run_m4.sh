#!/usr/bin/env bash
# M4's training runs, then the evaluation (decision 40 in docs/PLAN.md). A run that already
# finished (runs/m4/<run>/best.pt) is skipped, so the script can be started again after a stop.
# About six hours on the M2 MacBook: the benchmark's timings, plus validation and LayoutLMv3 in parts.
#   caffeinate bash scripts/run_m4.sh          (caffeinate keeps the Mac awake until it ends)
# Run it with the environment active (source .venv/bin/activate), after scripts/prepare_m4.py.
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  local name="$1"
  shift
  if [ -f "runs/m4/$name/best.pt" ]; then
    echo "$name: finished already"
    return
  fi
  python scripts/train_kie.py --run "$name" "$@"
}

for seed in 0 1 2; do                                   # the three models, seed by seed
  run "roberta_word_true_s$seed" --model roberta --seed "$seed"
  run "lilt_word_true_s$seed" --model lilt --seed "$seed"
  run "layoutlmv3_word_true_s$seed" --model layoutlmv3 --seed "$seed"
done
run lilt_segment_true_s0 --model lilt --boxes segment --seed 0              # the papers' setting, as a check
run layoutlmv3_segment_true_s0 --model layoutlmv3 --boxes segment --seed 0
for seed in 0 1 2; do                                   # LiLT trained on OCR words
  run "lilt_word_ocr_s$seed" --model lilt --train-on ocr --seed "$seed"
done
python scripts/eval_kie.py
echo "M4 runs done: docs/results/m4_fields.md"
