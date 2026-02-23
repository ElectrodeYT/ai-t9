# ai-t9 GPU trainer image
#
# Build:
#   bash scripts/build_image.sh          # tags as ai-t9-trainer:latest
#   docker build -t ai-t9-trainer:latest .
#
# Use with vast_orchestrate.py:
#   python scripts/vast_orchestrate.py configs/vast-large.yaml \
#       --image ai-t9-trainer:latest --install skip
#
# Push to a registry before using on Vast.ai:
#   docker tag ai-t9-trainer:latest <registry>/<user>/ai-t9-trainer:latest
#   docker push <registry>/<user>/ai-t9-trainer:latest
#
# The image is intentionally *not* pushed automatically. Build it locally
# whenever the dependencies or source change, then push manually.

# ---- Base: PyTorch with CUDA -----------------------------------------------
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Minimal system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# ---- Layer-caching trick: install all PyPI deps first ----------------------
# Copy only the build metadata + a stub package so pip can resolve deps.
# This layer is re-used on rebuilds unless pyproject.toml changes.
COPY pyproject.toml README.md ./
RUN mkdir -p src/ai_t9 \
    && printf '__version__ = "0.0.0"\n' > src/ai_t9/__init__.py \
    && pip install --no-cache-dir ".[train,data]" \
    && pip uninstall -y ai-t9

# ---- Install the actual package --------------------------------------------
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# ---- Entrypoint ------------------------------------------------------------
# Pass a config path as CMD: docker run ai-t9-trainer /root/train_config.yaml
ENTRYPOINT ["ai-t9-run"]
