# %% [markdown]
# # INT8 KV Cache Lab — 学习 / 开发 Colab（Runtime A）
#
# 这份 notebook 用来 **学习动态 INT8 KV**。每个代码 cell 都可以单独跑：会自己找仓库、把包装进**当前内核**，并用 `sys.executable`（不要用可能不一致的 `!python`）。
#
# **Runtime 要求**
#
# - 本文件：GPU **≥ 24 GiB** 才能跑 Qwen PPL。合成 INT8（第 3 步）不强制 24 GiB。
# - **不要安装 vLLM**。Capture 必须另开 A100，打开 `notebooks/colab_vllm_capture.ipynb`。
#
# **包名说明：** `int8_kvcache_lab` 不是 Colab 预装库，源码在仓库的 `src/` 下，需要 `pip install -e .`。

# %%
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Lyr-GW/int8-kvcache-lab.git"
BRANCH = "main"
PROJECT = Path("/content/project")

if (PROJECT / "src" / "int8_kvcache_lab").is_dir():
    print("reuse existing clone", PROJECT)
else:
    if PROJECT.exists():
        raise SystemExit(f"{PROJECT} 已存在但不是本仓库，请改名。不要 rm -rf，以免打断正在进行的模型下载。")
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(PROJECT)])
os.chdir(PROJECT)
print("cwd", Path.cwd())

# %% [markdown]
# ## 1. 安装学习环境（不跑 7B，不装 vLLM）
#
# 卸载可能冲突的 `torchvision`，再用**当前内核**的 interpreter 做 editable install。
# VRAM 不足时只警告，仍允许跑第 2–3 步。

# %%
import runpy
import subprocess
import sys
from pathlib import Path

import torch


def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先跑 clone cell，目标目录是 /content/project。")


ROOT = _bootstrap_lab(with_deps=True)

if not torch.cuda.is_available():
    print("警告: 没有 CUDA。第 3 步可在 CPU 跑，PPL / benchmark 需要 GPU。")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU={props.name}  VRAM={props.total_memory / 1024**3:.1f} GiB  CUDA={torch.version.cuda}")
    if props.total_memory < 24 * 1024**3:
        print("警告: Qwen2.5-7B PPL 需要 ≥24 GiB。第 5 步会失败，先做第 2–3 步。")

if subprocess.run([sys.executable, "-m", "pip", "show", "torchvision"], capture_output=True).returncode == 0:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"])

print("install ok", sys.executable, "cwd", ROOT)

# %% [markdown]
# ## 2. CPU 单测
#
# 不加载 Qwen。必须用当前内核的 `sys.executable`，`!python` 在部分 Colab 镜像上不是同一个环境。

# %%
import runpy
import sys
from pathlib import Path

def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先跑 clone cell。")

ROOT = _bootstrap_lab(with_deps=True)
runpy.run_path(str(ROOT / "scripts" / "colab_runtime.py"))["run_python"](
    "-m", "pytest", "-q",
    "tests/test_quantization.py",
    "tests/test_paged_attention.py",
    "tests/test_evaluation.py",
    "tests/test_kv_analysis.py",
)

# %% [markdown]
# ## 3. 合成数据走一遍动态 INT8（本实验的核心）
#
# 对应代码：`quantization.py`、`paged_cache.py`、`attention.py`。
# Q 为 per-tensor scale，K/V 为 per-head scale。FP cache 仍常驻。
# 可在 Colab 直接跑；缺包时会自动 `pip install -e --no-deps`，不会删模型缓存。

# %%
import math
import runpy
from pathlib import Path

import torch


def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先 clone 到 /content/project。")


ROOT = _bootstrap_lab()

from int8_kvcache_lab import PagedKVCache, QuantConfig, paged_attention_dynamic_int8, paged_attention_reference
from int8_kvcache_lab.quantization import dequantize_int8, per_head_scale, per_tensor_scale, quantize_symmetric_int8

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
# 每步含 findmax + 全 cache 量化 + 临时分配，不是 fused INT8 kernel。

# %%
import runpy
from pathlib import Path

def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先 clone 到 /content/project。")

ROOT = _bootstrap_lab()
runpy.run_path(str(ROOT / "scripts" / "colab_runtime.py"))["run_python"](
    "benchmarks/benchmark_dynamic.py", "--batch", "4", "--seq-len", "1024", "--repeat", "50"
)

# %% [markdown]
# ## 5. 官方 Qwen PPL（唯一允许的模型评估 cell）
#
# 不要手写 teacher-forced 循环。官方 `_losses` 会传完整 prefix mask 和 `cache_position`。

# %%
import runpy
from pathlib import Path

def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先 clone 到 /content/project。")

ROOT = _bootstrap_lab()
from int8_kvcache_lab.evaluation import evaluate

status = evaluate(
    "Qwen/Qwen2.5-7B-Instruct",
    samples=4,
    context=128,
    output_dir=str(ROOT / "artifacts"),
)
print("evaluate_status", status, "(0=pass, 2=PPL gate failed)")

# %% [markdown]
# ## 6. 读本 runtime 已收集的数据
#
# 在仓库根目录的 `artifacts/` 下查找，不依赖你当前 Colab 的 cwd。

# %%
import json
import runpy
from pathlib import Path

def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先 clone 到 /content/project。")

ROOT = _bootstrap_lab()
reports = sorted((ROOT / "artifacts").glob("run-*.json"))
if not reports:
    print("no artifacts yet under", ROOT / "artifacts")
for path in reports:
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
# ## 7. （可选）单步 adapter
#
# 第 5 步已成功则跳过。本 cell 自备 `torch` import，不依赖前面格子的变量。

# %%
import runpy
from pathlib import Path

import torch

def _bootstrap_lab(*, with_deps: bool = False):
    for root in (Path("/content/project"), Path("/content/project/int8-kvcache-lab"), Path.cwd(), Path.cwd().parent):
        helper = root / "scripts" / "colab_runtime.py"
        if helper.is_file():
            return runpy.run_path(str(helper))["ensure_lab"](with_deps=with_deps)
    raise FileNotFoundError("未找到仓库。请先 clone 到 /content/project。")

ROOT = _bootstrap_lab()
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
print("next_token", tokenizer.decode(int(decode.logits[:, -1].argmax(dim=-1).item())))
print("layers_patched", len(adapter._original_forwards))
del model, adapter, prefill, decode
torch.cuda.empty_cache()

# %% [markdown]
# ## 8. Runtime B（全新会话，不要接在本 notebook 后面）
#
# `Runtime → Disconnect and delete runtime`，换 **A100**，打开
# `notebooks/colab_vllm_capture.ipynb`。
#
# 必须带走的数据：真实 KV `.pt`、四粒度 `relative_l2` / `scale_bytes`、K/V percentile、operator pytest。
# 标定**不会改** Runtime A 的 per-head 路径。
