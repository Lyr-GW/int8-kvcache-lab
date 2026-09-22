# %% [markdown]
# # INT8 KV Cache Lab — vLLM 0.29 stages 1 to 5
#
# Run this in a **fresh GPU runtime**. It installs the vLLM 0.29.0 wheel,
# clones the matching source tree for reading, and does not rebuild vLLM.
# The live paged-attention call is `FlashAttentionImpl.forward` ->
# `flash_attn_varlen_func`. Ampere prints `torch.ops._vllm_fa2_C.varlen_fwd`.
#
# Default model is Qwen2.5-1.5B with `max_model_len=256`. Switch `MODEL` to
# `Qwen/Qwen2.5-7B-Instruct` only on a 24 GiB or larger GPU.

# %%
REPO_URL = "https://github.com/Lyr-GW/int8-kvcache-lab.git"
# This branch contains the vLLM 0.29 flow. Switch back to main after it merges.
BRANCH = "cursor/latest-vllm-int8-flow-3602"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# %%
!git clone --branch {BRANCH} {REPO_URL} /content/project
%cd /content/project
!bash scripts/install_vllm_capture.sh

# %% [markdown]
# ## Stage 1 — eager baseline and the real call site
#
# Flags: `enforce_eager=True`, `enable_chunked_prefill=False`,
# `enable_prefix_caching=False`, `max_model_len=256`, backend `FLASH_ATTN`.

# %%
!python -m int8_kvcache_lab.vllm_probe \
    --model {MODEL} \
    --max-model-len 256 \
    --max-tokens 8

# %% [markdown]
# ## Stage 2 — capture one AIME-style prompt and choose the KV scale
#
# The artifact stores Q, K, V, the referenced paged KV, `block_table`,
# `slot_mapping`, `cu_seqlens_q`, `seqused_k`, and the kernel output.
# `per_channel` is one scale per `(kv_head, head_dim)`. A head whose channels
# differ by 8x or more selects that layout. A large absmax/p99.9 ratio selects
# the percentile scale.

# %%
!python -m int8_kvcache_lab.vllm_capture \
    --model {MODEL} \
    --dataset aime \
    --max-model-len 256 \
    --max-tokens 16 \
    --output artifacts/vllm-qwen-kv-cache.pt
!python -m int8_kvcache_lab.kv_analysis \
    --capture artifacts/vllm-qwen-kv-cache.pt \
    --relative-l2-target 0.01

# %% [markdown]
# ## Stage 3 — PyTorch reference versus the captured kernel, then a live swap
#
# The first command replays the dump on CPU. The pytest calls the installed
# `flash_attn_varlen_func`. The compare command generates one prompt four
# times: native FA, PyTorch FP, dequant INT8, and simulated INT8 GEMM.

# %%
!python -m int8_kvcache_lab.replay_capture \
    --capture artifacts/vllm-qwen-kv-cache.pt
!python -m pytest -q tests/test_vllm_operator.py tests/test_int8_paths.py tests/test_varlen_reference.py
!python -m int8_kvcache_lab.vllm_compare \
    --model {MODEL} \
    --granularity per_channel \
    --max-model-len 256 \
    --max-tokens 8

# %% [markdown]
# ## Stage 4 and 5 — two INT8 implementations and the score table
#
# A dequantizes K/V through FP32 and runs the unchanged FP attention.
# B folds per-channel K scales into Q, quantizes Q per tensor, and simulates
# the INT8 matmul in FP32. Softmax stays in FP32.
# The scorecard is a four-item smoke set (AIME, HumanEval, GPQA, MMLU style).
# Prefill stays FP. Only batch-1 decode is quantized.

# %%
!python -m int8_kvcache_lab.scorecard \
    --model {MODEL} \
    --dataset micro \
    --max-new-tokens 32
!ls -lh artifacts
