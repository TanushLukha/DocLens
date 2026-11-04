#!/usr/bin/env bash
# The checks of scripts/m5/probe.py, each part in the environment it needs (about 15 minutes).
#   bash scripts/m5/probe.sh 2>&1 | tee runs/m5_probe.log
# Run it from the main environment (source .venv/bin/activate), after scripts/setup_m5.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs/m5_probe

.venv_mlx/bin/mlx_vlm.server --host 127.0.0.1 --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6 \
  >> runs/m5_probe/mlx_server.log 2>&1 &
MLX=$!
trap 'kill "$MLX" 2>/dev/null || true' EXIT
for _ in $(seq 1 900); do
  curl -sf http://127.0.0.1:8111/health >/dev/null && break
  kill -0 "$MLX" 2>/dev/null || { echo "MLX VLM server stopped: see runs/m5_probe/mlx_server.log"; exit 1; }
  sleep 2
done
echo "== PaddleOCR VL's model through MLX VLM"
.venv_paddle/bin/python scripts/m5/probe.py vl_server
kill "$MLX" 2>/dev/null || true
wait "$MLX" 2>/dev/null || true

echo "== the same model through Transformers"
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/m5/probe.py vl_hf
echo "== PaddleOCR's page pipeline with the detector's input limited"
.venv_paddle/bin/python scripts/m5/probe.py paddle_det
python scripts/m5/probe.py report
