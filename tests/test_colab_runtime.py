import runpy
from pathlib import Path


def test_find_repo_root_uses_this_checkout(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    helper = repo / "scripts" / "colab_runtime.py"
    runtime = runpy.run_path(str(helper))
    monkeypatch.chdir(tmp_path)
    assert runtime["find_repo_root"]() == repo
    assert (runtime["find_repo_root"]() / "src" / "int8_kvcache_lab").is_dir()
