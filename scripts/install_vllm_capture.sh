#!/usr/bin/env bash
set -euo pipefail

# Install the ABI-matched vLLM runtime for the optional capture/operator stage.
# Run in a fresh Colab runtime: vLLM 0.6.6 pins Torch 2.5.1 and torchvision,
# whereas the dynamic-Qwen notebook intentionally uses a newer text-only stack.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_VERSION="${VLLM_VERSION:-0.6.6}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required. In Colab choose Runtime > Change runtime type > A100 GPU.")
print(f"Preparing a vLLM capture runtime on Python {sys.version.split()[0]}, CUDA {torch.version.cuda}.")
PY

python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall "vllm==${VLLM_VERSION}"
python -m pip install -e "${PROJECT_DIR}[dev]" --no-deps
python - <<'PY'
import importlib.metadata
import torch
assert importlib.metadata.version("vllm") == "0.6.6"
assert torch.cuda.is_available()
print(f"vLLM {importlib.metadata.version('vllm')} / torch {torch.__version__} installed")
PY
