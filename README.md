# Dynamic INT8 KV Cache Lab

Correctness-first experiments for dynamic INT8 KV cache on current vLLM.
The tested runtime is **vLLM 0.29.0**. On that release, paged KV attention is
not `PagedAttention.forward_decode`. It is:

```text
FlashAttentionImpl.forward
  -> flash_attn_varlen_func(..., block_table=..., seqused_k=...)
  -> torch.ops._vllm_fa2_C.varlen_fwd     # Ampere, including A100
  -> torch.ops._vllm_fa3_C.fwd            # Hopper
  -> vllm.vllm_flash_attn.cute            # Blackwell FA4
```

The lab cache layout remains `[num_blocks, block_size, 2, num_kv_heads, head_dim]`.
FlashAttention itself reads K and V as
`[num_blocks, block_size, num_kv_heads, head_dim]`. vLLM 0.29 stores the
packed cache as `[num_blocks, num_kv_heads, block_size, 2 * head_size]` and
splits it inside `FlashAttentionImpl.forward`.

`per_channel` means one scale per `(kv_head, head_dim)`, reduced over tokens.
That scale sits on the axis QK reduces, so the INT8 GEMM path folds it into Q.
`per_head` is one scale per KV head and can be multiplied outside the dot
product. Stage 2 chooses between them from the captured magnitudes. It does
not silently change the older per-head dynamic PPL path.

## Colab: stages 1–5 on vLLM 0.29.0

Use a fresh GPU runtime and `notebooks/colab_vllm_capture.ipynb`. The default
model is `Qwen/Qwen2.5-1.5B-Instruct` with `max_model_len=256`. Qwen2.5-7B
still needs about 24 GiB.

```bash
bash scripts/install_vllm_capture.sh
python -m int8_kvcache_lab.vllm_probe --max-model-len 256
python -m int8_kvcache_lab.vllm_capture --dataset aime --max-model-len 256
python -m int8_kvcache_lab.kv_analysis --capture artifacts/vllm-qwen-kv-cache.pt
python -m int8_kvcache_lab.replay_capture --capture artifacts/vllm-qwen-kv-cache.pt
python -m int8_kvcache_lab.vllm_compare --granularity per_channel --max-tokens 8
python -m int8_kvcache_lab.scorecard --dataset micro
```

| Stage | What it checks | Command |
|---|---|---|
| 1 | Eager generate, and the log names `FlashAttentionImpl.forward` plus the FA operator | `vllm_probe` |
| 2 | Dump Q, K, V, paged KV, block table, slot mapping, cu_seqlens, seqused_k, output. Recommend per-head or per-channel, and absmax or p99.9 | `vllm_capture`, `kv_analysis` |
| 3 | PyTorch varlen reference versus the captured kernel output, then versus a live `flash_attn_varlen_func` call | `replay_capture`, `pytest tests/test_vllm_operator.py`, `vllm_compare` |
| 4 | A: quantize K/V, dequantize through FP32, unchanged FP attention. B: INT8 QK/PV simulated in FP32, softmax in FP32 | `pytest tests/test_int8_paths.py` |
| 5 | Native FP, per-channel dequant, per-channel INT8 GEMM, and per-head dequant on a smoke set of AIME / HumanEval / GPQA / MMLU style items | `scorecard` |

`install_vllm_capture.sh` installs the v0.29.0 wheel, which already contains
the compiled FlashAttention operators, and clones the matching source tree to
`/content/vllm-source` for reading. `scripts/build_vllm_from_source.sh` is the
optional full CUDA rebuild. Colab often cannot finish that rebuild; the probe
still prints the live call site without it.

Stage 5 scores batch-1 decode. Prefill stays on the native FP kernel. The
INT8 path recomputes scales from the FP cache on every decode step. It is a
numerical check, not a resident INT8 cache and not an INT8 tensor-core kernel.
`--dataset huggingface` loads public AIME, MMLU, and GPQA prefixes.
HumanEval completions are stored and left unscored. The micro function item
is executed against its own calls.

Engine flags used by the probe and capture:

```text
enforce_eager=True
enable_chunked_prefill=False
enable_prefix_caching=False
max_model_len=256
attention_config.backend=FLASH_ATTN
```

## Separate Transformers PPL notebook

`notebooks/colab_int8_kvcache.ipynb` does not install vLLM. It runs the
original per-head dynamic path (Q per-tensor, K/V per-head, dequantize, then
FP attention) on `Qwen/Qwen2.5-7B-Instruct` and checks WikiText-2
teacher-forced PPL. Use a fresh runtime with at least 24 GiB. Do not continue
it in the vLLM 0.29 runtime: that wheel replaces Torch.

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m int8_kvcache_lab.evaluation --samples 4 --context 128
```

## Quality gates

- Stage 4A versus the FP reference, relative L2: `<= 5%`.
- Stage 4B versus stage 4A, relative L2: `<= 10%`.
- Captured kernel versus the PyTorch reference, relative L2: `<= 5%`.
- Live `flash_attn_varlen_func` versus the FP reference, relative L2: `<= 2%`
  when vLLM is installed.
- The older WikiText-2 PPL gate remains a relative increase `<= 1%` on the
  per-head dynamic path.

## Not in this tree

Static INT8 writes during prefill, a resident INT8 cache, and a real INT8
tensor-core kernel are still out of scope. Stage 4B accumulates the integer
values in FP32 so the scale algebra can be checked before writing that kernel.
