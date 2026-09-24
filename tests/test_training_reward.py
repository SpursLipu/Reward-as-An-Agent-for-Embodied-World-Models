"""Reward materiality and provenance checks; fixtures are not visual truth labels."""
import copy
import hashlib
import json

import pytest

from reward_as_agent.evidence_grounding import core_report
from reward_as_agent.evidence_schema import score_report
from reward_as_agent.task_contract import freeze_contract
from reward_as_agent.training_reward import (
    RESOLUTION_VERSION, SCORING_VERSION, apply_training_reward, failure_candidates,
    needs_failure_resolution, validate_failure_resolution,
)


def fixture(status="failed", confidence="high", target="match"):
    task = {"requested_action": "Place block in basket.", "target_description": "The block and basket.",
            "final_state_requirement": "Block rests inside basket."}
    source = "Place block in basket. Keep the left arm still."
    contract = freeze_contract(source, {
        "schema_version": "task-contract-v1", "task": task, "coverage_notes": [],
        "requirements": [{"id": "R1", "source_quote": "Place block in basket.", "text": "Place block in basket."},
                         {"id": "R2", "source_quote": "Keep the left arm still.", "text": "Keep the left arm still."}],
    })
    report = {"schema_version": "evidence-v2", "task": task,
              "observations": [{"id": "E1", "frames": [0, 1], "description": "The block remains on the table."}],
              "task_assessment": {"verdict": status, "confidence": confidence, "target_match": target,
                                  "evidence": ["E1"], "reason": "Placement not achieved."},
              "physics_assessment": {"verdict": "plausible", "confidence": "high", "evidence": ["E1"],
                                     "issues": [], "reason": "Visible states are plausible."},
              "visual_assessment": {"verdict": "clear", "confidence": "high", "evidence": ["E1"],
                                    "reason": "Relevant states are visible."},
              "uncertainties": [], "requirement_checks": [
                  {"requirement_id": "R1", "status": "not_met", "evidence": ["E1"],
                   "reason": "The block remains on the table."},
                  {"requirement_id": "R2", "status": "met", "evidence": ["E1"],
                   "reason": "The left arm does not move."},
              ]}
    return report, contract, audits(report)


def audits(report, issues=None):
    return [{"report_sha256": hashlib.sha256(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "audit": {"schema_version": "requirement-scope-audit-v1", "checks": [
            {"requirement_id": check["requirement_id"], "issues": (issues or {}).get(check["requirement_id"], [])}
            for check in report["requirement_checks"]]}}]


def resolution(requirement_id="R1", established=True):
    return {"schema_version": RESOLUTION_VERSION,
            "decision": "failure_established" if established else "unresolved",
            "requirement_id": requirement_id if established else None,
            "evidence": ["E1"] if established else [], "confidence": "high" if established else "low",
            "independent_of_unresolved": established, "reason": "Visible final placement is outside the basket.",
            "uncertainty_analysis": "The unresolved detail does not change the visible placement failure."}


def score(report, contract, scope, **kwargs):
    return apply_training_reward(score_report(core_report(report)), report, contract, scope, **kwargs)


def issue(quote, kind="added_condition", refs=None):
    return {"kind": kind, "claim_quote": quote, "reference_requirement_ids": refs or [],
            "reason": "The cited claim is not grounded in this requirement."}


@pytest.mark.parametrize("visual,diagnostic", [("minor_degradation", 0.28), ("clear", 0.3)])
def test_clear_task_failure_trains_as_zero_regardless_of_auxiliary_score(visual, diagnostic):
    report, contract, _ = fixture()
    report["visual_assessment"]["verdict"] = visual
    scope = audits(report)
    out = score(report, contract, scope)
    assert out["total_score"] == out["task_reward"] == 0.0
    assert out["diagnostic_score"] == diagnostic
    assert not out["review_required"]
    assert out["decisive_failure"]


def test_uncertain_target_and_unrelated_scope_require_independent_resolution():
    report, contract, _ = fixture(target="uncertain")
    report["visual_assessment"]["verdict"] = "minor_degradation"
    scope = audits(report, {"R2": [issue("left arm")]})
    assert failure_candidates(report, contract, scope) == ["R1"]
    assert needs_failure_resolution(report, contract, scope)
    before = score(report, contract, scope)
    assert before["task_reward"] is None and not before["decisive_failure"]
    after = score(report, contract, scope, failure_resolution=resolution())
    assert after["task_reward"] == 0 and after["diagnostic_review_required"]
    assert after["diagnostic_score"] == 0.28


def test_uncertain_target_gets_quality_zero_without_claiming_decisive_failure():
    report, contract, _ = fixture(target="uncertain")
    out = score(report, contract, audits(report), failure_resolution=resolution(established=False))
    assert not out["review_required"] and out["task_reward"] == 0
    assert not out["decisive_failure"]
    assert "task: target identity is uncertain" in out["video_quality_gate"]["reasons"]


@pytest.mark.parametrize("confidence", ["low", "medium"])
def test_non_high_confidence_failure_gets_quality_zero(confidence):
    report, contract, scope = fixture(confidence=confidence)
    out = score(report, contract, scope)
    assert not out["review_required"] and out["task_reward"] == 0
    assert out["total_score"] == 0 and out["diagnostic_score"] == 0.3
    assert not out["decisive_failure"]
    assert not needs_failure_resolution(report, contract, scope)


def test_scope_tainted_sole_witness_is_withheld():
    report, contract, _ = fixture()
    scope = audits(report, {"R1": [issue("block remains")]})
    assert failure_candidates(report, contract, scope) == []
    out = score(report, contract, scope)
    assert out["review_required"] and out["task_reward"] is None


def test_unrelated_scope_issue_requires_resolution_but_does_not_taint_witness():
    report, contract, _ = fixture()
    scope = audits(report, {"R2": [issue("left arm")]})
    assert failure_candidates(report, contract, scope) == ["R1"]
    assert needs_failure_resolution(report, contract, scope)
    assert score(report, contract, scope)["task_reward"] is None
    out = score(report, contract, scope, failure_resolution=resolution())
    assert out["task_reward"] == 0 and out["diagnostic_review_required"]


def test_unobservable_other_requirement_gets_zero_without_resolution():
    report, contract, _ = fixture()
    report["requirement_checks"][1].update(status="unobservable", evidence=[])
    scope = audits(report)
    assert needs_failure_resolution(report, contract, scope)
    assert score(report, contract, scope)["task_reward"] == 0
    assert score(report, contract, scope, failure_resolution=resolution())["task_reward"] == 0


def test_internal_contradiction_also_taints_referenced_witness():
    report, contract, _ = fixture()
    scope = audits(report, {"R2": [issue("left arm", "internal_contradiction", ["R1"])]})
    assert failure_candidates(report, contract, scope) == []
    with pytest.raises(ValueError, match="scope-clean"):
        validate_failure_resolution(resolution(), report, contract, scope, [0, 1])


def test_other_requirement_scope_does_not_taint_reference_itself():
    report, contract, _ = fixture()
    scope = audits(report, {"R2": [issue("left arm", "other_requirement_condition", ["R1"])]})
    assert failure_candidates(report, contract, scope) == ["R1"]


def test_contract_ambiguity_always_blocks_zero():
    report, contract, scope = fixture()
    out = score(report, contract, scope, contract_review={"reason": "Two incompatible meanings remain."},
                failure_resolution=resolution())
    assert out["task_reward"] is None and not out["decisive_failure"]
    assert any(reason.startswith("task contract:") for reason in out["review_reasons"])


def test_unobservable_input_gets_quality_zero():
    report, contract, scope = fixture()
    scoring = score_report(core_report(report))
    scoring["total_score"] = None
    out = apply_training_reward(scoring, report, contract, scope,
                                physics_evidence={"input_unobservable": True},
                                failure_resolution=resolution())
    assert out["task_reward"] == 0 and not out["decisive_failure"]
    assert out["video_quality_gate"]["applied"]
    assert out["quality_adjusted_diagnostic_score"] == 0


def test_protocol_or_required_external_evidence_failure_is_not_cleared():
    report, contract, scope = fixture()
    scoring = score_report(core_report(report))
    scoring["review_reasons"].append("required physical evidence: no external tool records")
    scoring["physical_completion"] = {"required": True, "complete": False}
    out = apply_training_reward(scoring, report, contract, scope)
    assert out["task_reward"] is None and out["review_reasons"] == scoring["review_reasons"]


def test_audit_hash_mismatch_rejected():
    report, contract, scope = fixture()
    scope[-1]["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audited report hash"):
        score(report, contract, scope)


def test_normal_partial_unchanged_and_inputs_not_mutated():
    report, contract, scope = fixture(status="partial")
    scoring = score_report(core_report(report))
    before = copy.deepcopy((scoring, report, contract, scope))
    out = apply_training_reward(scoring, report, contract, scope)
    assert (scoring, report, contract, scope) == before
    assert out["total_score"] == out["task_reward"] == out["diagnostic_score"] == 0.615
    assert not out["review_required"] and not out["decisive_failure"]
    assert out["scoring_version"] == SCORING_VERSION


def test_unmet_requirements_do_not_veto_observed_partial_progress():
    report, contract, _ = fixture(status="partial")
    # An atomic condition can remain unmet despite useful partial execution.
    # The reducer must not replace visual progress judgement with a count.
    for check in report["requirement_checks"]:
        check["status"] = "not_met"
    scope = audits(report)
    out = score(report, contract, scope)
    assert out["task_reward"] == 0.615
    assert not out["decisive_failure"]


def test_background_condition_met_does_not_rescue_failed_core_task():
    report, contract, scope = fixture()
    assert report["requirement_checks"][1]["status"] == "met"
    out = score(report, contract, scope)
    assert out["task_reward"] == 0.0
    assert out["decisive_failure"]


def test_diagnostic_score_is_base_reducer_not_optional_process_gate_result():
    report, contract, scope = fixture(status="partial")
    scoring = score_report(core_report(report))
    base_version = scoring["scoring_version"]
    scoring.update(total_score=0.1, scoring_version="optional-process-gate")
    out = apply_training_reward(scoring, report, contract, scope)
    assert out["total_score"] == out["task_reward"] == 0.1
    assert out["diagnostic_score"] == 0.615
    assert out["diagnostic_scoring_version"] == base_version


def test_auxiliary_physics_doubt_is_diagnostic_when_failure_is_decisive():
    report, contract, _ = fixture()
    report["physics_assessment"]["issues"] = [{
        "kind": "motion", "severity": "major", "certainty": "uncertain", "evidence": ["E1"],
        "reason": "Cause of movement is unclear.", "alternative_explanation": "Recording cadence may explain it.",
    }]
    scope = audits(report)
    assert not needs_failure_resolution(report, contract, scope)
    out = score(report, contract, scope)
    assert out["task_reward"] == 0 and out["diagnostic_review_required"]
    assert out["review_reasons"] == []


def test_resolution_rejects_unknown_evidence_or_unsupplied_frames():
    report, contract, scope = fixture()
    value = resolution()
    value["evidence"] = ["E99"]
    with pytest.raises(ValueError, match="shared"):
        validate_failure_resolution(value, report, contract, scope, [0, 1])
    with pytest.raises(ValueError, match="unsupplied"):
        validate_failure_resolution(resolution(), report, contract, scope, [0])


@pytest.mark.parametrize("field,value", [("confidence", "medium"), ("independent_of_unresolved", False),
                                         ("uncertainty_analysis", ""), ("evidence", [])])
def test_resolution_rejects_inadequate_independence(field, value):
    report, contract, scope = fixture()
    decision = resolution()
    decision[field] = value
    with pytest.raises(ValueError):
        validate_failure_resolution(decision, report, contract, scope, [0, 1])


def test_witness_without_shared_task_evidence_is_not_candidate():
    report, contract, _ = fixture()
    report["observations"].append({"id": "E2", "frames": [0], "description": "The left arm is visible."})
    report["task_assessment"]["evidence"] = ["E2"]
    scope = audits(report)
    assert failure_candidates(report, contract, scope) == []
    assert score(report, contract, scope)["task_reward"] == 0


@pytest.mark.parametrize("dimension", ["task", "physics", "visual"])
def test_unobservable_dimension_returns_numeric_zero_and_preserves_report(dimension):
    report, contract, _ = fixture(status="partial")
    report[dimension + "_assessment"].update(verdict="unobservable", confidence="low")
    before = copy.deepcopy(report)
    out = score(report, contract, audits(report))
    assert out["total_score"] == out["task_reward"] == 0
    assert not out["review_required"] and not out["decisive_failure"]
    assert out["video_quality_gate"]["applied"]
    assert out["quality_adjusted_diagnostic_score"] == 0
    assert out["diagnostic_score"] is None
    assert all("no numeric reward" not in reason for reason in out["video_quality_gate"]["reasons"])
    assert report == before


def test_unclear_video_does_not_hide_required_tool_failure():
    report, contract, _ = fixture(status="unobservable", confidence="low")
    scoring = score_report(core_report(report))
    scoring["physical_completion"] = {"required": True, "complete": False}
    scoring["review_reasons"].append("required physical evidence: failed tool")
    out = apply_training_reward(scoring, report, contract, audits(report))
    assert out["review_required"] and out["task_reward"] is None
    assert not out["video_quality_gate"]["applied"]


def test_visible_partial_with_unresolved_required_action_returns_zero():
    report, contract, _ = fixture(status="partial")
    report["requirement_checks"][0].update(status="uncertain", evidence=[])
    out = score(report, contract, audits(report))
    assert out["task_reward"] == 0 and not out["review_required"]
    assert out["diagnostic_score"] == 0.615
    assert out["quality_adjusted_diagnostic_score"] == 0
