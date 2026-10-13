from contextlib import contextmanager
from types import SimpleNamespace

from int8_kvcache_lab import vllm_scorecard


def test_groups_compare_native_dequant_and_int8_compute():
    groups = vllm_scorecard.evaluation_groups("per_channel", include_per_head=True)
    assert groups == [
        ("native_fp", None, "per_channel"),
        ("per_channel_int8_dequant", "dequant", "per_channel"),
        ("per_channel_int8_gemm", "int8_gemm", "per_channel"),
        ("per_head_int8_dequant", "dequant", "per_head"),
    ]


def test_score_table_shows_accuracy_and_denominators():
    table = vllm_scorecard.format_score_table(
        [
            {"suite": "aime", "group": "native_fp", "correct": 8, "total": 10, "accuracy": 0.8},
            {"suite": "aime", "group": "per_channel_int8_dequant", "correct": 7, "total": 10, "accuracy": 0.7},
        ],
        ["native_fp", "per_channel_int8_dequant"],
    )
    assert "AIME" in table
    assert "8/10 (80.0%)" in table
    assert "7/10 (70.0%)" in table


def test_humaneval_completion_runs_in_isolated_grader():
    item = {
        "kind": "humaneval",
        "prompt": "def max2(a, b):\n    \"\"\"Return the larger number.\"\"\"\n",
        "entry_point": "max2",
        "test": "def check(candidate):\n    assert candidate(3, 9) == 9\n    assert candidate(-4, -1) == -1\n",
    }
    assert vllm_scorecard.grade(item, "    return a if a > b else b\n")
    assert not vllm_scorecard.grade(item, "    return a\n")


def test_scorecard_quantizes_query_and_reports_three_groups(monkeypatch, tmp_path):
    import sys

    calls = []

    class FakeEngine:
        def generate(self, prompts, params):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="391")]) for _ in prompts]

    @contextmanager
    def fake_replace(implementation, **kwargs):
        calls.append((implementation, kwargs))
        kwargs["call_stats"]["quantized_pytorch_calls"].value = 1
        yield

    monkeypatch.setattr(vllm_scorecard, "replace_varlen_with_pytorch", fake_replace)
    monkeypatch.setattr(vllm_scorecard, "make_llm", lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr(vllm_scorecard, "write_report", lambda report, output: "scorecard.json")
    monkeypatch.setattr(
        "int8_kvcache_lab.vllm_runtime.configure_worker_multiprocessing",
        lambda method: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )

    report = vllm_scorecard.run_scorecard(
        model="fake", tasks=[{"suite": "aime", "prompt": "compute", "answer": "391", "kind": "integer"}],
        dtype="bfloat16", max_model_len=64, max_new_tokens=2, granularity="per_channel",
        statistic="absmax", include_per_head=False, output_dir=str(tmp_path),
    )

    assert [name for name, _, _ in vllm_scorecard.evaluation_groups("per_channel", False)] == report["groups"]
    assert all(kwargs["query_granularity"] == "per_head" for _, kwargs in calls)
    assert calls[0][0] == "dequant"
    assert calls[1][0] == "int8_gemm"
    assert all(row["correct"] for row in report["rows"])
