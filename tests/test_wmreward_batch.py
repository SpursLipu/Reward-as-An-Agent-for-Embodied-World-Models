"""Batch routing tests; no GPU or model requests."""
from unittest.mock import patch

from scripts import run_wmreward_demo as runner


def test_default_batch_and_failure_exit(tmp_path):
    seen = []

    async def fake_run(args, demo):
        seen.append(demo)
        return int(demo == "demo_02")

    with patch.object(runner, "run", fake_run):
        code = runner.main([
            "--output", str(tmp_path / "batch"),
            "--worker-python", "/unused/python", "--wmreward-repo", "/unused/repo",
            "--checkpoint", "/unused/model", "--checkpoint-sha256", "a" * 64,
        ])
    assert seen == list(runner.DEMOS)
    assert code == 1


def test_selected_demos_are_deduplicated(tmp_path):
    seen = []

    async def fake_run(args, demo):
        seen.append(demo)
        return 0

    with patch.object(runner, "run", fake_run):
        code = runner.main([
            "--output", str(tmp_path / "batch"),
            "--worker-python", "/unused/python", "--wmreward-repo", "/unused/repo",
            "--checkpoint", "/unused/model", "--checkpoint-sha256", "a" * 64,
            "--demo", "demo_02", "--demo", "demo_02",
        ])
    assert seen == ["demo_02"]
    assert code == 0
