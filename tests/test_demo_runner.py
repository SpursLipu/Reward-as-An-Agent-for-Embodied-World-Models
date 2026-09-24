"""Offline checks for bundled demo discovery and request validation."""

import json
from pathlib import Path

import pytest

from scripts import run_demos as runner


def make_case(root: Path, name: str = "demo_06") -> Path:
    directory = root / "examples" / name
    directory.mkdir(parents=True)
    prompt = "Place the cube in the bin.\n"
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    (directory / "video.mp4").write_bytes(b"test video input")
    (directory / "request.json").write_text(json.dumps({
        "video_path": [f"examples/{name}/video.mp4"],
        "prompt": prompt,
    }), encoding="utf-8")
    return directory


def test_discover_demos_includes_new_bundles_in_sorted_order(tmp_path):
    for name in ("demo_08", "demo_01", "demo_06", "demo_07", "demo_10"):
        make_case(tmp_path, name)
    assert runner.discover_demos(tmp_path) == (
        "demo_01", "demo_06", "demo_07", "demo_08", "demo_10",
    )


@pytest.mark.parametrize("name", ["demo_6", "demo_006", "demo_06_old", "demo_ab", "other_06"])
def test_discovery_ignores_nonstandard_names(tmp_path, name):
    make_case(tmp_path, name)
    assert runner.discover_demos(tmp_path) == ()


@pytest.mark.parametrize("missing", ["request.json", "prompt.txt"])
def test_discovery_ignores_incomplete_input_bundles(tmp_path, missing):
    directory = make_case(tmp_path)
    (directory / missing).unlink()
    assert runner.discover_demos(tmp_path) == ()


def test_discovery_ignores_files_and_missing_examples(tmp_path):
    assert runner.discover_demos(tmp_path) == ()
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "demo_06").write_text("not a directory", encoding="utf-8")
    assert runner.discover_demos(tmp_path) == ()


def test_shared_demos_import_remains_compatible():
    from scripts import run_wmreward_demo

    assert isinstance(runner.DEMOS, tuple)
    assert run_wmreward_demo.DEMOS is runner.DEMOS
    assert set(f"demo_{index:02d}" for index in range(1, 6)) <= set(runner.DEMOS)


def test_main_refreshes_discovery_without_contacting_service(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    make_case(tmp_path, "demo_08")
    with pytest.raises(SystemExit) as exc:
        runner.main(["--help"])
    assert exc.value.code == 0
    assert "--demo {demo_08}" in capsys.readouterr().out


def test_main_rejects_empty_bundle_set_before_service_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--output", str(tmp_path / "output")])
    assert exc.value.code == 2
    assert "no bundled demos" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("value", [
    "http://127.0.0.1:7024", "http://127.0.0.1:7024/", "http://127.0.0.1:7024/eval_video/",
])
def test_service_urls_accepts_base_or_evaluation_endpoint(value):
    assert runner.service_urls(value) == (
        "http://127.0.0.1:7024/health", "http://127.0.0.1:7024/eval_video",
    )


@pytest.mark.parametrize("value", [
    "file:///tmp/service", "http://user:password@example.com", "https://example.com?token=x",
    "https://example.com#fragment",
])
def test_service_urls_rejects_invalid_or_sensitive_urls(value):
    with pytest.raises(ValueError):
        runner.service_urls(value)


def test_load_case_preserves_original_prompt_and_requests_details(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    directory = make_case(tmp_path)
    payload, inputs = runner.load_case("demo_06")
    assert payload == {
        "video_path": [str(directory / "video.mp4")],
        "prompt": "Place the cube in the bin.\n",
        "return_details": True,
    }
    assert inputs == [directory / "request.json", directory / "prompt.txt", directory / "video.mp4"]


def test_load_case_rejects_prompt_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    directory = make_case(tmp_path)
    (directory / "prompt.txt").write_text("Edited task", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        runner.load_case("demo_06")


@pytest.mark.parametrize("videos", [[], ["one.mp4", "two.mp4"], "one.mp4", ["missing.mp4"]])
def test_load_case_rejects_invalid_video_inputs(tmp_path, monkeypatch, videos):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    directory = make_case(tmp_path)
    path = directory / "request.json"
    request = json.loads(path.read_text(encoding="utf-8"))
    request["video_path"] = videos
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one video|existing repository file"):
        runner.load_case("demo_06")


def test_load_case_rejects_video_outside_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    monkeypatch.setattr(runner, "REPO_ROOT", root)
    directory = make_case(root)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"external video")
    path = directory / "request.json"
    request = json.loads(path.read_text(encoding="utf-8"))
    request["video_path"] = [str(outside)]
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="existing repository file"):
        runner.load_case("demo_06")


def test_motion_validation_accepts_real_low_survival():
    runner.validate_motion_evidence([{
        "tool": "cotracker3-motion-v1", "status": "insufficient_tracks",
        "model_provenance": {"checkpoint_sha256": "abc"},
        "source_decoded_sha256": "def", "tracking_source_frames": [0, 1],
    }])


@pytest.mark.parametrize("records", [[], [{"tool": "disabled", "status": "disabled"}],
    [{"tool": "cotracker3-motion-v1", "status": "ok"}]])
def test_motion_validation_rejects_missing_execution(records):
    with pytest.raises(ValueError):
        runner.validate_motion_evidence(records)
