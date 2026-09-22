#!/usr/bin/env bash
set -euo pipefail

# Optional source build. Colab disk and time limits often cannot finish a
# full vLLM CUDA compile. The default learning path uses the v0.29.0 wheel,
# which already ships the FlashAttention operators, and reads this checkout
# to locate FlashAttentionImpl.forward.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${PROJECT_DIR}/configs/versions.env" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/configs/versions.env"
fi
VLLM_DIR="${VLLM_DIR:-/content/vllm-source}"
if [[ ! -f "${VLLM_DIR}/pyproject.toml" && ! -f "${VLLM_DIR}/setup.py" ]]; then
  echo "Clone the source first with scripts/install_vllm_capture.sh" >&2
  exit 1
fi
python -m pip install -e "${VLLM_DIR}" --no-build-isolation
python - <<'PY'
import importlib.metadata
print("rebuilt vLLM", importlib.metadata.version("vllm"))
PY
