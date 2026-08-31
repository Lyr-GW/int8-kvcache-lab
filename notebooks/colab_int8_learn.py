# %% [markdown]
# # INT8 KV Cache Lab — 学习 / 开发 Colab（Runtime A）
#
# 这份 notebook 用来 **学习动态 INT8 KV**，不是再写一遍 FP16 NaN 排查循环。
#
# **Runtime 要求**
#
# - 本文件：GPU **≥ 24 GiB**（L4 / A100）。**不要安装 vLLM**。
# - Capture / 标定 / operator oracle：必须 **Disconnect and delete runtime** 后，另开 A100，打开 `notebooks/colab_vllm_capture.ipynb`。
#
# **相对仓库官方评估脚本的差别**
#
# - 不先跑 `bootstrap_colab.sh` 全流程（它会立刻 pytest + benchmark + 7B PPL）。
# - 按「算子 → 合成 attention → 官方 PPL → 读报告」学习。
# - 禁止手写 teacher-forced 循环；PPL 只走 `int8_kvcache_lab.evaluation`。
#
# ---
#
# ## 你当前 notebook 要怎么改
#
# | 现有内容 | 处理 |
# |---|---|
# | `fix_script` / 替换 `git checkout 8df14cfc8` | **整段删除**。当前 `scripts/bootstrap_colab.sh` 没有这行。 |
# | 对 `configs/versions.env` 做同样替换 | **删除**。 |
# | 三份几乎相同的 WikiText decode 循环（查 `FIRST_NONFINITE`） | **全部删除**。它们没装 adapter，也没传 `cache_position`，进不了 INT8。 |
# | `%%bash` + 再 load 一遍 7B | **删除**。模型只应加载一次。 |
# | clone 后 `%cd /content/project/int8-kvcache-lab` | **改成** `%cd /content/project`。仓库根目录就是 lab。 |
# | 最后的 `git pull` + `evaluation` | 保留评估思想，但改用下面的官方 cell；装包用本文件的 install cell。 |
#
# 改完应只剩：clone → install → 合成实验 / 测试 → benchmark → 官方 evaluation → 读 `artifacts`。

# %%
import os
import shutil

REPO_URL = "https://github.com/Lyr-GW/int8-kvcache-lab.git"
BRANCH = "main"
PROJECT = "/content/project"

if os.path.exists(PROJECT):
    shutil.rmtree(PROJECT)

# Colab magics: clone into the repo root (this lab is not nested).
!git clone --depth 1 --branch {BRANCH} {REPO_URL} {PROJECT}
%cd /content/project

# %% [markdown]
# ## 1. 安装学习环境（不跑 7B，不装 vLLM）
#
# 只做：CUDA / VRAM 检查、卸掉可能冲突的 `torchvision`、`pip install -e .[dev]`。
# vLLM 源码对照可选，学习 INT8 不必 checkout。

# %%
import torch

if not torch.cuda.is_available():
    raise SystemExit("需要 GPU。Runtime → Change runtime type → L4/A100。")
props = torch.cuda.get_device_properties(0)
print(f"GPU={props.name}  VRAM={props.total_memory / 1024**3:.1f} GiB  CUDA={torch.version.cuda}")
if props.total_memory < 24 * 1024**3:
    raise SystemExit("Qwen2.5-7B PPL 需要 ≥24 GiB。换 L4/A100，或先只跑下面的合成实验。")

import subprocess, sys
if subprocess.run([sys.executable, "-m", "pip", "show", "torchvision"], capture_output=True).returncode == 0:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
print("install ok", sys.executable)

# %% [markdown]
# ## 2. CPU 单测：先确认量化 / paged attention 契约
#
# 这些测试不加载 Qwen。失败的话后面的 PPL 没有意义。

# %%
!python -m pytest -q tests/test_quantization.py tests/test_paged_attention.py tests/test_evaluation.py tests/test_kv_analysis.py

# %% [markdown]
# ## 3. 合成数据走一遍动态 INT8（本实验的核心）
#
# 对应代码：
#
# - `quantization.py`：`per_tensor_scale(Q)`，`per_head_scale(K/V)` + `valid_mask`
# - `paged_cache.py`：物理布局 `[blocks, block_size, 2, kv_heads, head_dim]`
# - `attention.py`：全量 absmax → 临时 INT8 cache → dequant 再算 attention
#
# 动态路径 **不会** 让 FP cache 消失；INT8 是每步临时的。因此总显存不会减半。

# %%
import math
import torch
from int8_kvcache_lab import PagedKVCache, QuantConfig, paged_attention_dynamic_int8, paged_attention_reference
from int8_kvcache_lab.quantization import per_head_scale, per_tensor_scale, quantize_symmetric_int8, dequantize_int8

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch, seq_len, block_size = 1, 33, 16
q_heads, kv_heads, dim = 8, 2, 16
pages = math.ceil(seq_len / block_size)

cache = PagedKVCache.empty(pages, block_size, kv_heads, dim, dtype=torch.float16, device=device)
slots = torch.arange(seq_len, device=device)
cache.write(
    torch.randn(seq_len, kv_heads, dim, device=device, dtype=torch.float16),
    torch.randn(seq_len, kv_heads, dim, device=device, dtype=torch.float16),
    slots,
)
# 在最后一个 page 塞一个极大值，但 valid_mask 不应覆盖它。
cache.values[-1, -1, 0, 0, 0] = 999.0

query = torch.randn(batch, q_heads, dim, device=device, dtype=torch.float16)
block_tables = torch.arange(pages, device=device).view(1, pages)
seq_lens = torch.tensor([seq_len], device=device, dtype=torch.long)
config = QuantConfig(block_size=block_size)

fp_out = paged_attention_reference(query, cache, block_tables, seq_lens)
int8_out, stats = paged_attention_dynamic_int8(query, cache, block_tables, seq_lens, config)
rel_l2 = (int8_out.float() - fp_out.float()).norm() / fp_out.float().norm().clamp_min(1e-8)

valid = cache.valid_mask(block_tables, seq_lens)
key_scale = per_head_scale(cache.values[:, :, 0], valid)
q_scale = per_tensor_scale(query)
print("layout", tuple(cache.values.shape))
print("q_scale", q_scale.tolist())
print("key_scale", key_scale.tolist())
print("value_scale", stats["value_scale"].tolist())
print("attention_rel_l2", float(rel_l2))
print("fp_cache_bytes", stats["fp_cache_bytes"])
print("int8_cache_bytes", stats["int8_cache_bytes"])
print("scale_bytes", stats["scale_bytes"])
print("valid_slots", int(valid.sum()), "physical_slots", valid.numel())
assert float(rel_l2) <= 0.05
assert key_scale[0] < 999 / 127, "invalid page slot leaked into per-head absmax"

q_int8 = quantize_symmetric_int8(query, q_scale[:, None, None])
q_back = dequantize_int8(q_int8, q_scale[:, None, None], query.dtype)
print("query_roundtrip_rel_l2", float((q_back.float() - query.float()).norm() / query.float().norm().clamp_min(1e-8)))

# %% [markdown]
# ## 4. 合成 benchmark（预期 INT8 更慢）
#
# 报告字段：`attention_relative_l2_error`、`fp16_ms`、`dynamic_int8_ms`、cache/scale 字节。
# 慢是因为每步 findmax + 全 cache 量化 + 临时分配，不是 fused INT8 kernel。

# %%
!python benchmarks/benchmark_dynamic.py --batch 4 --seq-len 1024 --repeat 50

# %% [markdown]
# ## 5. 官方 Qwen PPL（唯一允许的模型评估 cell）
#
# **不要**再写 `model(input_ids=...)` 手搓循环。官方 `_losses` 会传：
#
# - 完整 prefix `attention_mask`
# - 精确 `cache_position`
# - 先 baseline，再 `QwenDynamicKVAdapter.install` 后的 candidate
#
# 质量门：candidate PPL 相对 baseline 涨幅 `<= 1%`。
# dtype 由 `_select_model_dtype()` 选择（支持则 BF16），不要写死 FP16。

# %%
from int8_kvcache_lab.evaluation import evaluate

status = evaluate(
    "Qwen/Qwen2.5-7B-Instruct",
    samples=4,
    context=128,
    output_dir="artifacts",
)
print("evaluate_status", status, "(0=pass, 2=PPL gate failed)")

# %% [markdown]
# ## 6. 读本 runtime 已收集的数据
#
# 应能看到两类 JSON：
#
# - `kind=dynamic_int8_benchmark`
# - `kind=qwen_dynamic_int8_ppl`（含 `decode_diagnostics`：logit 误差、greedy 是否一致）

# %%
import json
from pathlib import Path

for path in sorted(Path("artifacts").glob("run-*.json")):
    data = json.loads(path.read_text())
    kind = data.get("kind")
    print("=" * 60)
    print(path.name, kind)
    if kind == "dynamic_int8_benchmark":
        print("attention_relative_l2_error", data.get("attention_relative_l2_error"))
        print("fp16_ms", data.get("fp16_ms"))
        print("dynamic_int8_ms", data.get("dynamic_int8_ms"))
        quant = data.get("quantization") or {}
        for key in ("fp_cache_bytes", "int8_cache_bytes", "scale_bytes", "int8_cache_and_scale_bytes"):
            print(key, quant.get(key))
    elif kind == "qwen_dynamic_int8_ppl":
        print("baseline_ppl", data.get("baseline_ppl"))
        print("candidate_ppl", data.get("candidate_ppl"))
        print("relative_ppl_change", data.get("relative_ppl_change"))
        print("passes_strict_gate", data.get("passes_strict_gate"))
        for row in data.get("decode_diagnostics") or []:
            print(
                "diag",
                "match=", row.get("greedy_exact_match"),
                "logit_mae=", row.get("logit_max_abs_error"),
                "logit_rel_l2=", row.get("logit_relative_l2_error"),
            )

# %% [markdown]
# ## 7. （可选）单步 adapter：确认 INT8 路径真的被调用
#
# 仅在跳过第 5 步、或想确认 adapter 门控真的 patch 了各层时运行。会再加载 7B。
# 第 5 步已成功则跳过。当前 adapter 丢弃 attention stats；scale 请看第 3 步合成实验。
# 门控：`batch=1`、`seq=1`、`past_key_value` 非空，否则仍走原版 attention。

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from int8_kvcache_lab.evaluation import _select_model_dtype
from int8_kvcache_lab.qwen_adapter import QwenDynamicKVAdapter

device = torch.device("cuda")
dtype = _select_model_dtype()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=dtype,
    attn_implementation="eager",
).to(device).eval()

prompt = "Explain why KV cache helps autoregressive decoding in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
prompt_len = input_ids.shape[1]
prefill_mask = torch.ones((1, prompt_len), dtype=torch.long, device=device)
prefill_pos = torch.arange(prompt_len, dtype=torch.long, device=device)

with torch.inference_mode():
    prefill = model(
        input_ids=input_ids,
        attention_mask=prefill_mask,
        cache_position=prefill_pos,
        use_cache=True,
    )
    forced = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    adapter = QwenDynamicKVAdapter()
    adapter.install(model)
    decode = model(
        input_ids=forced,
        attention_mask=torch.ones((1, prompt_len + 1), dtype=torch.long, device=device),
        past_key_values=prefill.past_key_values,
        cache_position=torch.tensor([prompt_len], dtype=torch.long, device=device),
        use_cache=True,
    )
print("adapter_decode_finite", bool(torch.isfinite(decode.logits).all()))
print("next_token", tokenizer.decode(decode.logits[:, -1].argmax(dim=-1).item()))
print("layers_patched", len(adapter._original_forwards))
del model, adapter, prefill, decode
torch.cuda.empty_cache()

# %% [markdown]
# ## 8. Runtime B（全新会话，不要接在本 notebook 后面）
#
# `Runtime → Disconnect and delete runtime`，换 **A100**，打开
# `notebooks/colab_vllm_capture.ipynb`。
#
# 必须带走的数据：
#
# | 文件 | 看什么 |
# |---|---|
# | `artifacts/vllm-qwen-kv-cache.pt` | 每层逻辑 KV `[tokens, 2, kv_heads, head_dim]` |
# | calibration JSON `recommendation` | 最便宜且 rel-L2≤1% 的粒度；**不会改** Runtime A 的 per-head 路径 |
# | `candidates.*.relative_l2` / `scale_bytes_fp32` | per_tensor / per_head / per_token / per_channel 的误差与 scale 开销 |
# | `distribution.key/value.percentiles` | p50/p90/p99/p999/absmax，解释 outlier |
# | operator pytest | vLLM FP vs lab FP（≤2%）vs dynamic INT8（相对 native ≤5%） |
#
# 标定报告只指导「静态量化以后可能选哪种粒度」。当前 dynamic 路径仍然是 **Q per-tensor、K/V per-head、每 decode 步临时 INT8**。
