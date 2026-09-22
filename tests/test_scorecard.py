from int8_kvcache_lab.scorecard import MICRO_TASKS, grade, summarize


def test_micro_grades_cover_the_four_stage5_suites():
    assert {item["suite"] for item in MICRO_TASKS} == {"aime", "humaneval", "gpqa", "mmlu"}
    assert grade(MICRO_TASKS[0], "The product is 391.")
    assert not grade(MICRO_TASKS[0], "I think it is 17.")
    assert grade(MICRO_TASKS[2], "Answer: B")
    assert grade(MICRO_TASKS[3], "B. Paris")
    source = "def max2(a, b):\n    return a if a > b else b\n"
    assert grade(MICRO_TASKS[1], source)
    assert grade({"kind": "function", "answer": "max2", "calls": ()}, source) is None


def test_summary_reports_accuracy_per_group():
    rows = [
        {"suite": "aime", "group": "native_fp", "correct": True},
        {"suite": "aime", "group": "native_fp", "correct": False},
        {"suite": "aime", "group": "per_channel_dequant", "correct": None},
    ]
    summary = summarize(rows)
    native = next(row for row in summary if row["group"] == "native_fp")
    dequant = next(row for row in summary if row["group"] == "per_channel_dequant")
    assert native["accuracy"] == 0.5
    assert dequant["accuracy"] is None
    assert dequant["unscored"] == 1
