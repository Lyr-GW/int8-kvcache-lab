"""Reproducible JSON reporting for Colab runs."""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import shlex
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    packages: dict[str, str | None] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("vllm", "transformers", "datasets", "flash-attn", "int8-kvcache-lab"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
    except ImportError:
        packages = {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def default_output_dir() -> Path:
    """Return the active run archive directory, or the legacy artifacts folder."""
    return Path(os.environ.get("INT8_KVCACHE_OUTPUT_DIR", "artifacts"))


def _source_revision() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _snapshot_source(directory: Path) -> dict[str, Any]:
    """Preserve the in-flight source delta so dirty-run results remain inspectable."""
    root_text = None
    try:
        root_text = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False}
    root = Path(root_text)
    paths = ["README.md", "pyproject.toml", "src", "tests", "scripts", "notebooks"]
    try:
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *paths], cwd=root
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        patch = b""
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    patch_path = source_dir / "working-tree.patch"
    patch_path.write_bytes(patch)
    copied: list[str] = []
    try:
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "tests"],
            cwd=root,
            text=True,
        ).splitlines()
        for relative in untracked:
            candidate = Path(relative)
            if candidate.suffix != ".py" or any(part in {"__pycache__", ".pytest_cache"} for part in candidate.parts):
                continue
            source = root / candidate
            if source.is_file():
                destination = source_dir / "untracked" / candidate
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(directory)))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "available": True,
        "base_commit": _source_revision()["commit"],
        "patch_path": str(patch_path.relative_to(directory)),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "patch_size_bytes": len(patch),
        "untracked_python_files": copied,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_archive(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create or load the run manifest; safe to call from every stage process."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if metadata:
            manifest.setdefault("metadata", {}).update(metadata)
            _write_json_atomic(manifest_path, manifest)
        return manifest_path
    encoded_metadata = os.environ.get("INT8_KVCACHE_RUN_METADATA")
    if metadata is None and encoded_metadata:
        try:
            metadata = json.loads(encoded_metadata)
        except json.JSONDecodeError:
            metadata = {"metadata_parse_error": encoded_metadata}
    source_snapshot = _snapshot_source(directory)
    source_artifacts = []
    if source_snapshot.get("available"):
        patch_path = directory / source_snapshot["patch_path"]
        source_artifacts.append({
            "path": source_snapshot["patch_path"],
            "kind": "source_patch",
            "size_bytes": patch_path.stat().st_size,
            "sha256": source_snapshot["patch_sha256"],
        })
        for relative in source_snapshot["untracked_python_files"]:
            path = directory / relative
            source_artifacts.append({
                "path": relative,
                "kind": "untracked_source_snapshot",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest = {
        "schema": "int8-kvcache-lab.experiment.v1",
        "run_id": run_id or os.environ.get("INT8_KVCACHE_RUN_ID") or f"adhoc-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "metadata": metadata or {},
        "source": _source_revision(),
        "source_snapshot": source_snapshot,
        "environment": environment(),
        "stages": {},
        "artifacts": source_artifacts,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _update_manifest(output_dir: str | Path, update) -> dict[str, Any]:
    directory = Path(output_dir)
    path = initialize_archive(directory)
    manifest = json.loads(path.read_text())
    update(manifest)
    _write_json_atomic(path, manifest)
    return manifest


def register_artifact(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    kind: str = "file",
    details: dict[str, Any] | None = None,
) -> None:
    """Add a produced file to the experiment manifest (idempotently)."""
    artifact = Path(path)
    if not artifact.exists():
        return
    root = Path(output_dir) if output_dir is not None else default_output_dir()
    try:
        stored_path = str(artifact.resolve().relative_to(root.resolve()))
    except ValueError:
        stored_path = str(artifact.resolve())
    entry = {
        "path": stored_path,
        "kind": kind,
        "size_bytes": artifact.stat().st_size,
        "sha256": _file_sha256(artifact),
        **(details or {}),
    }

    def add(manifest: dict[str, Any]) -> None:
        artifacts = manifest.setdefault("artifacts", [])
        artifacts[:] = [item for item in artifacts if item.get("path") != stored_path]
        artifacts.append(entry)

    _update_manifest(root, add)


def write_report(payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Write a JSON artifact, including runtime environment details.

    Every value goes through :func:`to_jsonable`, so a caller that leaves a
    tensor or a numpy scalar in the payload still gets a usable report.
    """
    directory = Path(output_dir) if output_dir is not None else default_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = initialize_archive(directory)
    manifest = json.loads(manifest_path.read_text())
    document = {
        "run_id": manifest["run_id"],
        "environment": environment(),
        "source": manifest["source"],
        **payload,
    }
    stage = os.environ.get("INT8_KVCACHE_STAGE")
    if stage:
        document.setdefault("stage", stage)
    path = directory / f"{payload.get('experiment', payload.get('kind', 'report'))}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    _write_json_atomic(path, document)
    register_artifact(path, output_dir=directory, kind="report")
    return path


def run_stage(stage: str, command: list[str], output_dir: str | Path) -> int:
    """Run a stage command, stream+save its complete log, and record exit status."""
    if not command:
        raise ValueError("run_stage requires a command after --")
    directory = Path(output_dir)
    initialize_archive(directory)
    log_dir = directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = "".join(character if character.isalnum() or character in "-_" else "_" for character in stage)
    log_path = log_dir / f"{safe_stage}.log"
    started = datetime.now(timezone.utc).isoformat()
    environment_vars = os.environ.copy()
    environment_vars.update({
        "INT8_KVCACHE_RUN_ID": json.loads((directory / "manifest.json").read_text())["run_id"],
        "INT8_KVCACHE_OUTPUT_DIR": str(directory.resolve()),
        "INT8_KVCACHE_STAGE": stage,
    })
    print(f"[{stage}] $ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment_vars)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    finished = datetime.now(timezone.utc).isoformat()
    exit_code = int(process.wait())
    relative_log = str(log_path.relative_to(directory))

    def record(manifest: dict[str, Any]) -> None:
        manifest.setdefault("stages", {})[stage] = {
            "status": "passed" if exit_code == 0 else "failed",
            "command": command,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "exit_code": exit_code,
            "log": relative_log,
        }
        artifacts = manifest.setdefault("artifacts", [])
        artifacts[:] = [item for item in artifacts if item.get("path") != relative_log]
        artifacts.append({
            "path": relative_log,
            "kind": "stage_log",
            "size_bytes": log_path.stat().st_size,
            "sha256": _file_sha256(log_path),
        })

    _update_manifest(directory, record)
    if stage == "stage2_capture" and exit_code != 0:
        for partial in directory.glob("*.worker.pt"):
            register_artifact(partial, output_dir=directory, kind="partial_worker_capture")
    return exit_code


def finalize_archive(output_dir: str | Path, *, status: str = "complete") -> Path:
    """Write a human-readable summary and set the terminal run status."""
    directory = Path(output_dir)
    current = json.loads(initialize_archive(directory).read_text())
    required = current.get("metadata", {}).get("required_stages", [])
    stage_states = current.get("stages", {})
    if status == "complete":
        if any(stage_states.get(name, {}).get("status") == "failed" for name in required):
            status = "failed"
        elif any(stage_states.get(name, {}).get("status") != "passed" for name in required):
            status = "incomplete"

    def finish(manifest: dict[str, Any]) -> None:
        manifest["status"] = status
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    manifest = _update_manifest(directory, finish)
    lines = [
        f"# INT8 KV Cache experiment: {manifest['run_id']}",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Started (UTC): {manifest['started_at_utc']}",
        f"- Finished (UTC): {manifest['finished_at_utc']}",
        f"- Git commit: `{manifest['source'].get('commit')}` (dirty: `{manifest['source'].get('dirty')}`)",
        f"- Metadata: `{json.dumps(manifest.get('metadata', {}), sort_keys=True)}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Exit | Log |",
        "|---|---|---:|---|",
    ]
    for name, info in manifest.get("stages", {}).items():
        lines.append(f"| {name} | {info.get('status')} | {info.get('exit_code')} | [{info.get('log')}]({info.get('log')}) |")
    lines.extend(["", "## Results at a glance", ""])
    reports = []
    for item in manifest.get("artifacts", []):
        if item.get("kind") != "report":
            continue
        report_path = directory / item["path"]
        if report_path.exists():
            try:
                reports.append(json.loads(report_path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
    for report in reports:
        if "recommendation" in report:
            recommendation = report["recommendation"]
            lines.extend([
                "### Stage 2 calibration",
                "",
                f"- KV granularity: `{recommendation.get('kv_granularity')}`",
                f"- Scale statistic: `{recommendation.get('scale_statistic')}`",
                f"- Meets reconstruction target: `{recommendation.get('meets_target')}`",
                "",
            ])
        if report.get("experiment") == "stage3_replay":
            lines.extend([
                "### Stage 3 replay",
                "",
                f"- Max relative L2: `{report.get('max_relative_l2')}`",
                f"- Stage-3 gate passed: `{report.get('passes_stage3_gate')}`",
                "",
            ])
        if report.get("experiment") == "stage4_capture_numerics":
            lines.extend(["### Stage 4 captured-input numerics", ""])
            lines.append(f"- Quality gates: `{json.dumps(report.get('quality_gates', {}), sort_keys=True)}`")
            lines.append("")
            for name, result in report.get("comparisons", {}).items():
                lines.append(
                    f"- `{name}`: max rel-L2 vs native `{result.get('max_relative_l2_vs_native')}`, "
                    f"vs dequant FP-Q `{result.get('max_relative_l2_vs_dequant_q_fp')}`"
                )
            lines.append("")
        if report.get("kind") == "stage5_vllm_dynamic_int8_scorecard":
            lines.extend([
                f"### Stage 5 scorecard — Q `{report.get('query_granularity')}`, KV `{report.get('kv_granularity')}`",
                "",
                report.get("score_table_markdown", "(score table missing)"),
                "",
            ])
    lines.extend(["", "## Artifacts", ""])
    for item in manifest.get("artifacts", []):
        lines.append(f"- `{item.get('kind')}`: [{item.get('path')}]({item.get('path')}) ({item.get('size_bytes')} bytes)")
    lines.extend(["", "Full structured index: [manifest.json](manifest.json).", ""])
    summary_path = directory / "README.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    register_artifact(summary_path, output_dir=directory, kind="summary")
    return summary_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manage reproducible experiment archives.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output-dir", type=Path, required=True)
    init_parser.add_argument("--run-id")
    init_parser.add_argument("--metadata", default="{}")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--stage", required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    finish_parser = subparsers.add_parser("finalize")
    finish_parser.add_argument("--output-dir", type=Path, required=True)
    finish_parser.add_argument("--status", choices=("complete", "incomplete", "failed"), default="complete")
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output-dir", type=Path, required=True)
    register_parser.add_argument("--path", type=Path, required=True)
    register_parser.add_argument("--kind", default="file")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        manifest = initialize_archive(args.output_dir, run_id=args.run_id, metadata=json.loads(args.metadata))
        print(manifest.parent)
    elif args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        raise SystemExit(run_stage(args.stage, command, args.output_dir))
    elif args.action == "finalize":
        print(finalize_archive(args.output_dir, status=args.status))
    elif args.action == "register":
        register_artifact(args.path, output_dir=args.output_dir, kind=args.kind)
    else:
        directory = args.output_dir
        zip_path = args.zip or directory.with_suffix(".zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in directory.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(directory.parent))
        print(zip_path)


if __name__ == "__main__":
    main()
