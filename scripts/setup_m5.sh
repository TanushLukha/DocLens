#!/usr/bin/env bash
# M5's environments, once per machine (decision 43 in docs/PLAN.md). Every system gets its own,
# because their pinned dependencies clash: Surya wants Pillow below 11 and one exact OpenCV, MLX VLM
# a newer OpenCV, PaddleOCR its own framework. Models download on first use (about 6 GB in all).
#   bash scripts/setup_m5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3.13}"

brew list llama.cpp >/dev/null 2>&1 || brew install llama.cpp      # Surya runs its model in llama.cpp's server

make_env() {                                 # name, then the packages it pins
  local name="$1"
  shift
  [ -x ".venv_$name/bin/python" ] || "$PY" -m venv ".venv_$name"
  ".venv_$name/bin/python" -m pip install -q --upgrade pip
  ".venv_$name/bin/python" -m pip install -q "$@"
  echo "$name: ready"
}

make_env paddle paddlepaddle==3.3.1 "paddleocr[doc-parser]==3.7.0"   # PaddleOCR, and the PaddleOCR VL pipeline
make_env easyocr easyocr==1.7.2
make_env surya surya_ocr==0.22.1
make_env mlx mlx_vlm==0.7.11                                         # serves PaddleOCR VL's model on the GPU
echo "M5 environments ready: .venv_paddle .venv_easyocr .venv_surya .venv_mlx"
