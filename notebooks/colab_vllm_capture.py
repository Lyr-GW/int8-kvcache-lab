# %% [markdown]
# # INT8 KV Cache Lab — vLLM Capture and PagedAttention Oracle
#
# Run this in a **fresh A100 Colab runtime**.  It installs vLLM 0.6.6, whose
# Torch ABI differs from the dynamic-Qwen notebook.  This notebook performs
# stage 1 (real decode KV capture + calibration) and stage 2 (native vLLM
# PagedAttention versus the PyTorch INT8 reference).  It does not run PPL.

# %%
REPO_URL = "https://github.com/Lyr-GW/int8-kvcache-lab.git"
BRANCH = "main"

# %%
# In a fresh runtime this target must not already exist.
!git clone --branch {BRANCH} {REPO_URL} /content/project
%cd /content/project
!bash scripts/install_vllm_capture.sh

# %%
# A real Qwen decode.  The capture contains every layer's logical, valid K/V
# values only, never unused preallocated cache pages.
!python -m int8_kvcache_lab.vllm_capture \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-tokens 16 \
    --output artifacts/vllm-qwen-kv-cache.pt

# %%
# Compare per-tensor/head/token/channel reconstruction error and scale cost.
!python -m int8_kvcache_lab.kv_analysis \
    --capture artifacts/vllm-qwen-kv-cache.pt \
    --relative-l2-target 0.01

# %%
# This is the requested stage-2 CUDA operator test.  It invokes vLLM's actual
# PagedAttention forward_decode kernel and checks FP and dynamic INT8 paths.
!python -m pytest -q tests/test_vllm_operator.py

# %%
!ls -lh artifacts
