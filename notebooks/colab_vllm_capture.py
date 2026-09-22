# %% [markdown]
# # INT8 KV Cache Lab — vLLM Capture and PagedAttention Oracle
#
# Run this in a **fresh A100 Colab runtime**. It installs vLLM 0.6.6, whose
# Torch ABI differs from the dynamic-Qwen notebook. Do not continue a Runtime A
# session. Each code cell chdirs into the clone so a skipped `%cd` cannot hide
# `ModuleNotFoundError`.

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Lyr-GW/int8-kvcache-lab.git"
BRANCH = "main"
PROJECT = Path("/content/project")
if (PROJECT / "src" / "int8_kvcache_lab").is_dir():
    print("reuse existing clone", PROJECT)
else:
    if PROJECT.exists():
        raise SystemExit(f"{PROJECT} exists but is not this lab; rename it instead of rm -rf")
    subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO_URL, str(PROJECT)])
os.chdir(PROJECT)
subprocess.check_call(["bash", str(PROJECT / "scripts" / "install_vllm_capture.sh")])
print("python", sys.executable)

# %%
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/project")
os.chdir(PROJECT)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "int8_kvcache_lab.vllm_capture",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--max-tokens",
        "16",
        "--output",
        "artifacts/vllm-qwen-kv-cache.pt",
    ]
)

# %%
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/project")
os.chdir(PROJECT)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "int8_kvcache_lab.kv_analysis",
        "--capture",
        "artifacts/vllm-qwen-kv-cache.pt",
        "--relative-l2-target",
        "0.01",
    ]
)

# %%
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/project")
os.chdir(PROJECT)
subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests/test_vllm_operator.py"])
subprocess.check_call(["ls", "-lh", "artifacts"])
