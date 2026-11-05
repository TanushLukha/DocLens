# The DocLens service on the CPU (M6). Build it after scripts/export_models.py has filled models/:
#   docker build -t doclens .
#   docker run --rm -p 8000:8000 doclens          then open http://localhost:8000
# The image holds everything it needs: the code, the models and docTR's detector weights, so it
# reads documents without a network.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOCLENS_MODELS=/app/models DOCTR_CACHE_DIR=/app/doctr HF_HUB_OFFLINE=1
# OpenCV's wheel needs these two system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# the CPU build of PyTorch (docTR's detector and LiLT run in it), then the rest, pinned as on the Mac
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[service]"

# docTR's detector weights, fetched once at build time
RUN HF_HUB_OFFLINE=0 python -c "from doclens.doctr_lines import DoctrLines; DoctrLines(device='cpu')"
RUN useradd --create-home doclens && chown -R doclens /app/doctr
# the models are copied as the service's user: Transformers writes model.safetensors readable by its
# owner only (mode 600), and COPY keeps that mode with root as the owner, so the service could not
# read it (decision 55)
COPY --chown=doclens:doclens models ./models
USER doclens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "doclens.service:app", "--host", "0.0.0.0", "--port", "8000"]
