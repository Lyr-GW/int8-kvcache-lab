#!/usr/bin/env bash
set -euo pipefail

# This script is intended to run after this repository is cloned in /content.
# VLLM is checked out only for source-level comparison; it is not installed.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${PROJECT_DIR}/configs/versions.env" ]]; then
  # Shellcheck is intentionally not required in Colab; this file is static configuration.
  source "${PROJECT_DIR}/configs/versions.env"
fi
VLLM_DIR="${VLLM_DIR:-/content/vllm-source}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-refs/tags/v0.6.6}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required. In Colab choose Runtime > Change runtime type > T4/A100 GPU.")
properties = torch.cuda.get_device_properties(0)
minimum = 24 * 1024**3
print(f"GPU: {properties.name}; memory: {properties.total_memory / 1024**3:.1f} GiB; CUDA: {torch.version.cuda}")
if properties.total_memory < minimum:
    raise SystemExit("Qwen2.5-7B FP16 validation requires at least 24 GiB VRAM. Use an L4/A100 runtime.")
PY

python -m pip install --upgrade pip

# Qwen text evaluation does not use torchvision.  Colab images can contain a
# torchvision wheel built against a different Torch release; Transformers may
# import its optional image helpers and fail before the language model loads.
# Remove it before installing the project, while remaining harmless on images
# where torchvision is not present.
if python -m pip show torchvision >/dev/null 2>&1; then
  echo "Removing torchvision: it is unused here and can be incompatible with Colab's Torch build."
  python -m pip uninstall --yes torchvision
else
  echo "torchvision is not installed; no compatibility cleanup needed."
fi

python -m pip install -e "${PROJECT_DIR}[dev]"

if [[ ! -d "${VLLM_DIR}/.git" ]]; then
  git clone --filter=blob:none "${VLLM_REPO}" "${VLLM_DIR}"
fi
git -C "${VLLM_DIR}" fetch --depth=1 origin "${VLLM_REF}"
git -C "${VLLM_DIR}" checkout --detach FETCH_HEAD
echo "vLLM source reference: $(git -C "${VLLM_DIR}" rev-parse HEAD)"

cd "${PROJECT_DIR}"
python -m pytest -n 4 tests -q
python benchmarks/benchmark_dynamic.py
python -m int8_kvcache_lab.evaluation
