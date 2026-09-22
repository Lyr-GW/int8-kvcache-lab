"""Reproducible JSON reporting for Colab runs."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def to_jsonable(value: Any) -> Any:
    """Coerce a report value into something ``json.dumps`` can encode.

    A report is written at the end of a long GPU run, so one unexpected tensor
    must not cost the whole run. Tensors become a shape/dtype stub rather than
    being dropped silently, so a reader can still see what was elided.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # json.dumps would emit bare NaN/Infinity, which is not valid JSON.
        return value if math.isfinite(value) else str(value)
    if isinstance(value, torch.Tensor):
        stub: dict[str, Any] = {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
        if value.numel() == 1:
            stub["tensor_value"] = to_jsonable(value.item())
        return stub
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return to_jsonable(value.tolist())
    return repr(value)


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
    """Write a JSON artifact, including runtime environment details.

    Every value goes through :func:`to_jsonable`, so a caller that leaves a
    tensor or a numpy scalar in the payload still gets a usable report.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    document = {"environment": environment(), **payload}
    path = directory / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(to_jsonable(document), indent=2, sort_keys=True) + "\n")
    return path
