#!/usr/bin/env bash
# M5: every system reads the same exported images, then the scorer (decisions 43 to 48 in
# docs/PLAN.md). Images a system has already read are skipped, so the script can be started again.
#   bash scripts/run_m5.sh smoke            3 images of each set per system, shown: checks every runner
#   caffeinate bash scripts/run_m5.sh       everything, results to docs/results/m5_compare.md
# Run it with the main environment active (source .venv/bin/activate), after scripts/setup_m5.sh.
# A system that fails is reported at the end and the others still run; the scorer then compares
# the systems that read every image.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=()
if [ "${1:-}" = "smoke" ]; then
  export M5_RUNS=runs/m5_smoke                 # kept apart from the real results
  ARGS=(--limit 3 --show 3)
fi
mkdir -p "${M5_RUNS:-runs/m5}"
[ -f data/m5/export.json ] || python scripts/m5_export.py

FAILED=()
step() {                                       # a name, then the command that runs one system on one task
  local name="$1"
  shift
  echo "== $name"
  "$@" || { local code=$?; echo "!! $name failed with exit code $code; the other systems go on"; FAILED+=("$name"); }
}

MLX=""
mlx_up() {                                     # PaddleOCR VL's model, served on the Apple GPU
  .venv_mlx/bin/mlx_vlm.server --host 127.0.0.1 --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6 \
    >> "${M5_RUNS:-runs/m5}/mlx_server.log" 2>&1 &
  MLX=$!
  for _ in $(seq 1 900); do                    # the first start downloads the model (about 2 GB)
    curl -sf http://127.0.0.1:8111/health >/dev/null && return 0
    kill -0 "$MLX" 2>/dev/null || { echo "MLX VLM server stopped: see mlx_server.log"; return 1; }
    sleep 2
  done
  echo "MLX VLM server did not start in 30 minutes"
  return 1
}
mlx_down() {
  if [ -n "$MLX" ]; then
    kill "$MLX" 2>/dev/null || true
    wait "$MLX" 2>/dev/null || true
  fi
  MLX=""
}
trap mlx_down EXIT

paddleocr() {                                  # pages a set at a time, each in its own process (decision 48)
  if [ "$1" != "pages" ]; then
    .venv_paddle/bin/python scripts/m5/run_paddleocr.py --task "$1" ${ARGS[@]+"${ARGS[@]}"}
    return
  fi
  local code=0
  for part in synth cord; do
    .venv_paddle/bin/python scripts/m5/run_paddleocr.py --task pages --sets "$part" ${ARGS[@]+"${ARGS[@]}"} || code=$?
  done
  return "$code"
}

paddleocr_vl() {                               # the server up, the runner, the server down
  mlx_up || return 1
  local code=0
  .venv_paddle/bin/python scripts/m5/run_paddleocr_vl.py --task "$1" --server-pid "$MLX" ${ARGS[@]+"${ARGS[@]}"} || code=$?
  mlx_down
  return "$code"
}

for task in crops pages; do
  step "$task: DocLens" python scripts/m5/run_doclens.py --task "$task" ${ARGS[@]+"${ARGS[@]}"}
  step "$task: Tesseract" python scripts/m5/run_tesseract.py --task "$task" ${ARGS[@]+"${ARGS[@]}"}
  step "$task: PaddleOCR" paddleocr "$task"
  step "$task: EasyOCR" .venv_easyocr/bin/python scripts/m5/run_easyocr.py --task "$task" ${ARGS[@]+"${ARGS[@]}"}
  step "$task: PaddleOCR VL" paddleocr_vl "$task"
  step "$task: Surya" .venv_surya/bin/python scripts/m5/run_surya.py --task "$task" ${ARGS[@]+"${ARGS[@]}"}
done
step "crops: PaddleOCR VL's reference implementation, a part of the crops" \
  python scripts/m5/run_paddleocr_vl_ref.py --task crops ${ARGS[@]+"${ARGS[@]}"}
step "scoring" python scripts/m5_score.py
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
echo "M5 done"
