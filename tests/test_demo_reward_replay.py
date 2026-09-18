"""Offline migration never replaces a missing visual judgement or live inference."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from reward_as_agent.evidence_grounding import core_report, observation_time_manifest
from reward_as_agent.evidence_schema import score_report
from reward_as_agent.task_contract import freeze_contract, validate_requirement_checks
from scripts.replay_demo_reward import digest, replay


def example(target="match"):
    task = {"requested_action": "Place block in basket.", "target_description": "The block and basket.",
            "final_state_requirement": "Block rests inside basket."}
    contract = freeze_contract("Place block in basket.", {
        "schema_version": "task-contract-v1", "task": task, "coverage_notes": [],
        "requirements": [{"id": "R1", "source_quote": "Place block in basket.", "text": "Place block in basket."}],
    })
    report = {"schema_version": "evidence-v2", "task": task,
              "observations": [{"id": "E1", "frames": [0, 1], "description": "The block remains on the table."}],
              "task_assessment": {"verdict": "failed", "confidence": "high", "target_match": target,
                                  "evidence": ["E1"], "reason": "Placement not achieved."},
              "physics_assessment": {"verdict": "plausible", "confidence": "high", "evidence": ["E1"],
                                     "issues": [], "reason": "Visible states are plausible."},
              "visual_assessment": {"verdict": "clear", "confidence": "high", "evidence": ["E1"],
                                    "reason": "Relevant states are visible."},
              "uncertainties": [], "requirement_checks": [
                  {"requirement_id": "R1", "status": "not_met", "evidence": ["E1"],
                   "reason": "The block remains on the table."}]}
    audits = [{"report_sha256": hashlib.sha256(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "audit": {"schema_version": "requirement-scope-audit-v1", "checks": [
            {"requirement_id": "R1", "issues": []}]}}]
    scoring = score_report(core_report(report))
    scoring["scoring_version"] = "evidence-soft-physics-v2-with-scope-review-v5-provisional"
    manifest = [{"source_frame_index": 0, "timestamp_seconds": 0.0},
                {"source_frame_index": 1, "timestamp_seconds": 0.1}]
    details = {"scoring": scoring, "evidence_report": report, "task_contract": contract,
               "task_contract_sha256": contract["contract_sha256"], "evaluator_version": "original-live-evaluator",
               "video_metadata": {"fps": 10.0, "decoded_frames": 2}, "frame_manifest": manifest,
               "observation_time_manifest": observation_time_manifest(report, manifest),
               "requirement_summary": validate_requirement_checks(report["requirement_checks"], contract, report),
               "requirement_scope_audits": audits, "pre_scope_repair_report": copy.deepcopy(report),
               "planning_api_output": {"index": 0, "score": "failed", "status": "success"},
               "total_score": scoring["total_score"], "review_required": scoring["review_required"],
               "training_eligible": not scoring["review_required"],
               "trace": [{"stage": "verification", "output": copy.deepcopy(report), "response_id": "original-id"}]}
    return {"status": "needs_review" if scoring["review_required"] else "success",
            "score": None if scoring["review_required"] else scoring["total_score"], "index": 0,
            "details": details, "model": "recorded-model", "provider": "recorded-provider"}


def save_source(tmp_path, response=None, run=None):
    source = tmp_path / "source"
    source.mkdir()
    response = example() if response is None else response
    values = {"response.json": response,
              "run.json": {"execution_mode": "original-live", "evaluator_version": "original-live-evaluator",
                           "started_at": "original-start", "input_sha256": {}} if run is None else run,
              "tools.json": response["details"].get("physics_evidence"),
              "reports.json": {"before_tool_reflection": None,
                               "after_tool_reflection": response["details"]["pre_scope_repair_report"],
                               "final_after_scope_audit": response["details"]["evidence_report"]}}
    for name, value in values.items():
        (source / name).write_text(json.dumps(value))
    return source


def test_replay_preserves_evidence_bytes_original_metadata_and_source(tmp_path):
    source = save_source(tmp_path)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "output"
    summary = replay(source, output)
    assert summary["score"] == 0 and summary["original_score"] == 0.3
    assert summary["new_model_calls"] == summary["new_tool_calls"] == 0
    response = json.loads((output / "response.json").read_text())
    run = json.loads((output / "run.json").read_text())
    assert response["training_eligible"] is True
    assert response["diagnostic_score"] == 0.3
    assert response["details"]["trace"] == example()["details"]["trace"]
    assert response["details"]["evidence_report"] == example()["details"]["evidence_report"]
    assert response["details"]["evaluator_version"] == "original-live-evaluator"
    assert run["original_run"] == json.loads(original["run.json"])
    assert run["execution_mode"] == "offline_reward_policy_replay"
    policy = run["policy_replay"]
    assert policy["source_artifact_sha256"] == {name: digest(data) for name, data in original.items()}
    assert "reward_as_agent/training_reward.py" in policy["policy_source_sha256"]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    for name in ("tools.json", "reports.json"):
        assert (output / name).read_bytes() == original[name]
    for name in ("response.json", "run.json"):
        assert (output / ("source_" + name)).read_bytes() == original[name]


@pytest.mark.parametrize("kind", ["ambiguous", "frame", "contract", "audit", "score", "error", "resolution"])
def test_rejects_invalid_or_unresolved_sources_without_writing(tmp_path, kind):
    response = example(target="uncertain" if kind == "ambiguous" else "match")
    details = response["details"]
    if kind == "frame":
        details["frame_manifest"][1]["source_frame_index"] = 3
    elif kind == "contract":
        details["task_contract"]["source_sha256"] = "x"
    elif kind == "audit":
        details["requirement_scope_audits"][0]["report_sha256"] = "x"
    elif kind == "score":
        details["scoring"]["total_score"] = 0.9
    elif kind == "error":
        response["status"] = "error"
    elif kind == "resolution":
        details["failure_reward_resolution"] = {"decision": "failure_established"}
    source = save_source(tmp_path, response)
    with pytest.raises(ValueError):
        replay(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_refuses_overwrite_and_repeat_replay(tmp_path):
    source = save_source(tmp_path)
    output = tmp_path / "output"
    replay(source, output)
    before = (output / "response.json").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        replay(source, output)
    with pytest.raises(ValueError, match="Repeat"):
        replay(output, tmp_path / "second")
    assert (output / "response.json").read_bytes() == before


def test_changed_original_inputs_rejected(tmp_path):
    source = save_source(tmp_path, run={"input_sha256": {"README.md": "not-the-current-hash"}})
    with pytest.raises(ValueError, match="input hash mismatch"):
        replay(source, tmp_path / "output")


def test_valid_failure_certificate_still_requires_original_trace(tmp_path):
    response = example(target="uncertain")
    response["details"]["failure_reward_resolution"] = {
        "schema_version": "failure-reward-resolution-v1", "decision": "failure_established",
        "requirement_id": "R1", "evidence": ["E1"], "confidence": "high",
        "independent_of_unresolved": True, "reason": "Original visual check established failure.",
        "uncertainty_analysis": "The original check established target-independent absence of placement."}
    source = save_source(tmp_path, response)
    with pytest.raises(ValueError, match="original image-grounded trace"):
        replay(source, tmp_path / "output")
