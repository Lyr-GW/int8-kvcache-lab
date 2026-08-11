"""Reproducible JSON reporting for Colab runs."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def environment() -> dict[str, Any]:
    """Collect the runtime facts needed to interpret a performance result."""
    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            name=properties.name,
            memory_bytes=properties.total_memory,
            cuda=torch.version.cuda,
        )
    try:
        gpu["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu["nvidia_smi"] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": gpu,
    }


def write_report(payload: dict[str, Any], output_dir: str | Path = "artifacts") -> Path:
    """Write a JSON artifact, including runtime environment details."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    document = {"environment": environment(), **payload}
    path = directory / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path
