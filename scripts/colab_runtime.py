"""Colab helpers that locate this repo and install it into the current kernel.

This module must stay importable with only the stdlib. Notebook cells load it
with ``runpy.run_path`` so they can recover even when ``int8_kvcache_lab`` is
not on ``sys.path`` yet.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CANDIDATE_ROOTS = (
    Path("/content/project"),
    Path("/content/project/int8-kvcache-lab"),
)


def find_repo_root() -> Path:
    """Return the lab root that contains ``src/int8_kvcache_lab``."""
    here = Path(__file__).resolve().parent.parent
    ordered = (here, *CANDIDATE_ROOTS, Path.cwd(), Path.cwd().parent)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src" / "int8_kvcache_lab").is_dir():
            return resolved
    raise FileNotFoundError(
        "未找到仓库（需要 src/int8_kvcache_lab）。请先 clone 到 /content/project，不要 cd 到不存在的嵌套目录。"
    )


def ensure_lab(*, with_deps: bool = False) -> Path:
    """chdir into the repo, put ``src`` on ``sys.path``, and pip-install if needed.

    ``with_deps=False`` avoids re-resolving torch while a Qwen download is running.
    """
    root = find_repo_root()
    os.chdir(root)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import int8_kvcache_lab  # noqa: F401
    except ModuleNotFoundError:
        command = [sys.executable, "-m", "pip", "install", "-e", f"{root}[dev]"]
        if not with_deps:
            command.append("--no-deps")
        subprocess.check_call(command)
        import int8_kvcache_lab  # noqa: F401
    return root


def run_python(*args: str) -> None:
    """Run ``sys.executable`` in the repo root (not Colab's possibly-different ``python``)."""
    root = ensure_lab()
    if args[:2] == ("-m", "pytest"):
        try:
            import pytest  # noqa: F401
        except ModuleNotFoundError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest>=8", "pytest-xdist>=3.5"])
    subprocess.check_call([sys.executable, *args], cwd=root)
