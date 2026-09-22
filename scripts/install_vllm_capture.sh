#!/usr/bin/env bash
set -euo pipefail

# Install the tested vLLM wheel and clone the matching source tree.
# The wheel already contains the compiled FlashAttention operators
# (_vllm_fa2_C / _vllm_fa3_C / the FA4 package). Use
# scripts/build_vllm_from_source.sh only when you intend to rebuild those
# operators. Run this in a fresh Colab GPU runtime.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${PROJECT_DIR}/configs/versions.env" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/configs/versions.env"
fi
VLLM_VERSION="${VLLM_VERSION:-${VLLM_RUNTIME_VERSION:-0.29.0}}"
VLLM_DIR="${VLLM_DIR:-/content/vllm-source}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-refs/tags/v${VLLM_VERSION}}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU.")
print(f"Preparing vLLM on Python {sys.version.split()[0]}, CUDA {torch.version.cuda}.")
PY

python -m pip install --upgrade pip
python -m pip install --upgrade "vllm==${VLLM_VERSION}"
python -m pip install -e "${PROJECT_DIR}[dev]" --no-deps

if [[ ! -d "${VLLM_DIR}/.git" ]]; then
  git clone --filter=blob:none "${VLLM_REPO}" "${VLLM_DIR}"
fi
git -C "${VLLM_DIR}" fetch --depth=1 origin "${VLLM_REF}"
git -C "${VLLM_DIR}" checkout --detach FETCH_HEAD

python - <<PY
import importlib.metadata
import torch
version = importlib.metadata.version("vllm")
expected = "${VLLM_VERSION}"
if version != expected:
    raise SystemExit(f"expected vLLM {expected}, found {version}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after installing vLLM")
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.vllm_flash_attn import flash_attn_varlen_func
print(f"vLLM {version} / torch {torch.__version__}")
print(f"FlashAttentionImpl: {FlashAttentionImpl.__module__}")
print(f"flash_attn_varlen_func: {flash_attn_varlen_func.__module__}")
PY
echo "vLLM source: ${VLLM_DIR} @ $(git -C "${VLLM_DIR}" rev-parse --short HEAD)"
