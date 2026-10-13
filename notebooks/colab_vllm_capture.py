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
BRANCH = "codex/fix-colab-vllm-spawn"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# %%
from datetime import datetime, timezone
from pathlib import Path
import uuid

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
ARCHIVE_DIR = Path("/content/int8-kvcache-experiments") / RUN_ID
(ARCHIVE_DIR / "logs").mkdir(parents=True, exist_ok=True)
print(f"Experiment archive: {ARCHIVE_DIR}")

# %%
!if [ -d /content/project/.git ]; then git -C /content/project fetch origin {BRANCH} && git -C /content/project checkout --detach origin/{BRANCH}; else git clone --branch {BRANCH} {REPO_URL} /content/project; fi
%cd /content/project
!set -o pipefail && bash scripts/install_vllm_capture.sh 2>&1 | tee {ARCHIVE_DIR}/logs/environment_setup.log
!python -m int8_kvcache_lab.archive init --run-id {RUN_ID} --output-dir {ARCHIVE_DIR} --metadata '{{"model":"{MODEL}","notebook":"colab_vllm_capture.py","required_stages":["stage1_probe","stage2_capture","stage2_analysis","stage3_replay","stage3_live_alignment","stage3_operator_tests","stage4_capture_numerics","stage4_unit_tests","stage5_scorecard_q_per_head","stage5_scorecard_q_per_channel"]}}'
!python -m int8_kvcache_lab.archive register --output-dir {ARCHIVE_DIR} --path {ARCHIVE_DIR}/logs/environment_setup.log --kind setup_log

# %% [markdown]
# ## Stage 1 — eager baseline and the real call site
#
# Flags: `enforce_eager=True`, `enable_chunked_prefill=False`,
# `enable_prefix_caching=False`, `max_model_len=256`, backend `FLASH_ATTN`.

# %%
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage1_probe -- python -m int8_kvcache_lab.vllm_probe \
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
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage2_capture -- python -m int8_kvcache_lab.vllm_capture \
    --model {MODEL} \
    --dataset aime \
    --max-model-len 256 \
    --max-tokens 16 \
    --output {ARCHIVE_DIR}/vllm-qwen-kv-cache.pt
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage2_analysis -- python -m int8_kvcache_lab.kv_analysis \
    --capture {ARCHIVE_DIR}/vllm-qwen-kv-cache.pt \
    --output-dir {ARCHIVE_DIR} \
    --relative-l2-target 0.01

# %% [markdown]
# ## Stage 3 — PyTorch reference versus the captured kernel, then a live swap
#
# The first command replays the dump on CPU. The pytest calls the installed
# `flash_attn_varlen_func`. The compare command generates one prompt four
# times: native FA, PyTorch FP, dequant INT8, and simulated INT8 GEMM.

# %%
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage3_replay -- python -m int8_kvcache_lab.replay_capture \
    --capture {ARCHIVE_DIR}/vllm-qwen-kv-cache.pt \
    --output-dir {ARCHIVE_DIR}
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage3_live_alignment -- python -m int8_kvcache_lab.vllm_compare \
    --model {MODEL} \
    --granularity per_channel \
    --max-model-len 256 \
    --max-tokens 8 \
    --output-dir {ARCHIVE_DIR}
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage3_operator_tests -- python -m pytest -q --junitxml={ARCHIVE_DIR}/stage3-tests.xml tests/test_vllm_operator.py tests/test_varlen_reference.py
!python -m int8_kvcache_lab.archive register --output-dir {ARCHIVE_DIR} --path {ARCHIVE_DIR}/stage3-tests.xml --kind junit

# %% [markdown]
# ## Stage 4 — numerical INT8 attention checks
#
# Stage 4A quantizes K/V, dequantizes them and checks the FP attention math.
# Stage 4B simulates INT8 QK/PV in FP32. Stage 5 additionally quantizes Q
# dynamically per-head or per-channel before comparing full-model dataset scores.

# %%
stage2_reports = sorted(ARCHIVE_DIR.glob("vllm_kv_cache_calibration-*.json"))
if not stage2_reports:
    raise FileNotFoundError("Stage 2 calibration report is missing from the experiment archive")
STAGE2_REPORT = stage2_reports[-1]
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage4_capture_numerics -- python -m int8_kvcache_lab.stage4_validation \
    --capture {ARCHIVE_DIR}/vllm-qwen-kv-cache.pt \
    --analysis-report {STAGE2_REPORT} \
    --output-dir {ARCHIVE_DIR}
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage4_unit_tests -- python -m pytest -q --junitxml={ARCHIVE_DIR}/stage4-tests.xml tests/test_int8_paths.py tests/test_vllm_scorecard.py
!python -m int8_kvcache_lab.archive register --output-dir {ARCHIVE_DIR} --path {ARCHIVE_DIR}/stage4-tests.xml --kind junit

# %% [markdown]
# ## Stage 5 — end-to-end dynamic INT8 accuracy scorecard
#
# The same deterministic dataset samples run through both Q-scale choices:
# per-head and per-channel. Each scorecard compares native FP, Stage-2 KV
# granularity with dequantized attention math and INT8 QK/PV simulation; KV
# granularity/statistic come from the Stage-2 JSON.
# Optional per-head dequant is an additional comparison. The JSON and printed
# Markdown table include sample counts and scores for all four datasets.
# This is an accuracy prototype: it is not a resident INT8 cache or a speed test.

# %%
import json
from pathlib import Path

stage2_reports = []
for report_path in ARCHIVE_DIR.glob("vllm_kv_cache_calibration-*.json"):
    report = json.loads(report_path.read_text())
    if "recommendation" in report:
        stage2_reports.append(report_path)
stage2_reports.sort(key=lambda path: path.stat().st_mtime)
if not stage2_reports:
    raise FileNotFoundError("Run Stage 2 first; expected a JSON report under artifacts/")
STAGE2_REPORT = stage2_reports[-1]
print(f"Using Stage-2 scale recommendation: {STAGE2_REPORT}")
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage5_scorecard_q_per_head -- python -m int8_kvcache_lab.vllm_scorecard \
    --model {MODEL} \
    --dataset huggingface \
    --limit 100 \
    --max-model-len 4096 \
    --max-new-tokens 512 \
    --analysis-report {STAGE2_REPORT} \
    --query-granularity per_head \
    --include-per-head \
    --output-dir {ARCHIVE_DIR}
!python -m int8_kvcache_lab.archive run --output-dir {ARCHIVE_DIR} --stage stage5_scorecard_q_per_channel -- python -m int8_kvcache_lab.vllm_scorecard \
    --model {MODEL} \
    --dataset huggingface \
    --limit 100 \
    --max-model-len 4096 \
    --max-new-tokens 512 \
    --analysis-report {STAGE2_REPORT} \
    --query-granularity per_channel \
    --include-per-head \
    --output-dir {ARCHIVE_DIR}

# %% [markdown]
# ## Finalize and download the complete experiment archive
#
# The archive contains the source revision/environment manifest, command logs,
# raw Stage-2 capture, calibration histograms/scales, Stage-3 alignment output,
# Stage-4 numerical errors and test results, and Stage-5 prompts, answer keys,
# completions and score table. Keep the downloaded ZIP with the experiment.

# %%
!python -m int8_kvcache_lab.archive finalize --output-dir {ARCHIVE_DIR}
!python -m int8_kvcache_lab.archive export --output-dir {ARCHIVE_DIR} --zip {ARCHIVE_DIR}.zip
from google.colab import files
files.download(str(ARCHIVE_DIR) + ".zip")
