import json
import math
from pathlib import Path

import torch

from int8_kvcache_lab.reporting import to_jsonable, write_report


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
