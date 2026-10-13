import json
import math
from pathlib import Path

import torch

from int8_kvcache_lab.reporting import (
    finalize_archive,
    initialize_archive,
    register_artifact,
    run_stage,
    to_jsonable,
    write_report,
)


def test_tensors_become_shape_stubs_instead_of_raising():
    payload = {"block_tables": torch.zeros(2, 3, dtype=torch.long), "note": "ok"}
    encoded = to_jsonable(payload)
    assert encoded["note"] == "ok"
    assert encoded["block_tables"] == {"tensor_shape": [2, 3], "tensor_dtype": "torch.int64"}
    assert json.loads(json.dumps(encoded)) == encoded


def test_a_scalar_tensor_keeps_its_value():
    assert to_jsonable(torch.tensor(2.5))["tensor_value"] == 2.5


def test_nested_payloads_and_non_finite_floats_stay_encodable():
    encoded = to_jsonable({"steps": [{"layers": [torch.zeros(1)]}], "ratio": math.inf})
    # Bare Infinity is not valid JSON, so it is spelled out instead.
    assert encoded["ratio"] == "inf"
    assert encoded["steps"][0]["layers"][0]["tensor_shape"] == [1]
    assert json.loads(json.dumps(encoded)) == encoded


def test_write_report_survives_a_tensor_payload(tmp_path):
    path = write_report({"experiment": "test", "tensor": torch.ones(2)}, tmp_path)
    document = json.loads(Path(path).read_text())
    assert document["experiment"] == "test"
    assert document["tensor"]["tensor_shape"] == [2]
    assert "environment" in document


def test_archive_manifest_records_reports_artifacts_and_stage_logs(tmp_path):
    initialize_archive(
        tmp_path,
        run_id="test-run",
        metadata={"model": "fixture", "required_stages": ["stage3"]},
    )
    report_path = write_report({"experiment": "stage2_fixture", "score": 0.5}, tmp_path)
    extra = tmp_path / "capture.pt"
    extra.write_bytes(b"capture")
    register_artifact(extra, output_dir=tmp_path, kind="raw_capture")
    assert run_stage("stage3", ["python", "-c", "print('stage output')"], tmp_path) == 0
    summary = finalize_archive(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_id"] == "test-run"
    assert manifest["status"] == "complete"
    assert manifest["stages"]["stage3"]["status"] == "passed"
    assert "stage output" in (tmp_path / "logs/stage3.log").read_text()
    assert {artifact["kind"] for artifact in manifest["artifacts"]} >= {"report", "raw_capture", "stage_log", "summary"}
    assert "stage2_fixture" in report_path.name
    assert summary.exists()


def test_archive_is_marked_incomplete_when_a_required_stage_is_missing(tmp_path):
    initialize_archive(tmp_path, metadata={"required_stages": ["stage5"]})
    finalize_archive(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
