"""A zero alone cannot suppress the frozen physics-uncertainty diagnostic.

The fixtures are synthetic schema checks, not visual accuracy labels. No model,
network, video decoder, or external tool is called.
"""

import copy
import hashlib
import json
import unittest

from reward_as_agent.evidence_grounding import core_report, observation_time_manifest
from reward_as_agent.evidence_schema import score_report
from reward_as_agent.task_contract import freeze_contract, validate_requirement_checks
from reward_as_agent.training_reward import apply_training_reward, RESOLUTION_VERSION
from scripts.evidence.check_diagnostics import BLACK, JUMP, REAL, inspect_record, verified_failure_zero


FLAG = "uncertain_physics_issue_not_routed_to_review"


def fixture():
    task = {
        "requested_action": "Place block in basket.",
        "target_description": "Block and basket.",
        "final_state_requirement": "Block inside basket.",
    }
    source = "Place block in basket."
    contract = freeze_contract(source, {
        "schema_version": "task-contract-v1", "task": task,
        "requirements": [{"id": "R1", "source_quote": source, "text": source}],
        "coverage_notes": [],
    })
    report = {
        "schema_version": "evidence-v2", "task": copy.deepcopy(task),
        "observations": [{"id": "E1", "frames": [0, 1],
                          "description": "The block remains on the table outside the basket."}],
        "task_assessment": {
            "verdict": "failed", "confidence": "high", "evidence": ["E1"],
            "reason": "The visible block never enters the basket.", "target_match": "match",
        },
        "physics_assessment": {
            "verdict": "plausible", "confidence": "medium", "evidence": ["E1"],
            "reason": "The block remains supported, but a local motion ambiguity remains.",
            "issues": [{
                "kind": "motion", "severity": "minor", "certainty": "uncertain",
                "evidence": ["E1"], "reason": "A small change in the gripper is unclear.",
                "alternative_explanation": "Occlusion could explain the change.",
            }],
        },
        "visual_assessment": {
            "verdict": "clear", "confidence": "high", "evidence": ["E1"],
            "reason": "The block and basket are visible.",
        },
        "uncertainties": [],
        "requirement_checks": [{
            "requirement_id": "R1", "status": "not_met", "evidence": ["E1"],
            "reason": "The block remains outside the basket.",
        }],
    }
    report_hash = hashlib.sha256(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    audits = [{"report_sha256": report_hash, "audit": {
        "schema_version": "requirement-scope-audit-v1",
        "checks": [{"requirement_id": "R1", "issues": []}],
    }}]
    diagnostic = score_report(core_report(report))
    diagnostic["scoring_version"] = "evidence-soft-physics-v2-with-scope-review-v5-provisional"
    scoring = apply_training_reward(diagnostic, report, contract, audits)
    frames = [{"source_frame_index": i, "timestamp_seconds": i / 20, "view": "full"}
              for i in [0, 1]]
    details = {
        "evidence_report": report, "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "task_contract_registry_sha256": None, "task_contract_review": None,
        "requirement_summary": validate_requirement_checks(report["requirement_checks"], contract, report),
        "requirement_scope_audits": audits, "pre_scope_repair_report": copy.deepcopy(report),
        "frame_manifest": frames, "video_metadata": {"fps": 20, "decoded_frames": 2},
        "observation_time_manifest": observation_time_manifest(report, frames),
        "scoring": scoring, "total_score": scoring["total_score"],
        "review_required": scoring["review_required"],
        "training_eligible": scoring["total_score"] is not None and not scoring["review_required"],
        "physics_evidence": None, "failure_reward_resolution": None, "trace": [],
        **{key: scoring[key] for key in (
            "diagnostic_score", "diagnostic_review_required", "diagnostic_review_reasons",
        )},
    }
    spec = {"sample_id": sorted(REAL)[0], "prompt": source, "video_path": "fixture.mp4"}
    row = {**spec, "status": "success", "new_score": scoring["total_score"],
           "training_eligible": details["training_eligible"],
           "task_contract_sha256": contract["contract_sha256"], "details": details}
    return row, spec


class TrainingDiagnosticTests(unittest.TestCase):
    def test_verified_independent_failure_waives_only_physics_review_flag(self):
        row, spec = fixture()
        original = copy.deepcopy(row)
        self.assertTrue(verified_failure_zero(row, spec))
        result = inspect_record(row, spec)
        self.assertTrue(result["training_eligible"])
        self.assertEqual(result["score"], 0)
        self.assertNotIn(FLAG, result["structural_flags"])
        self.assertTrue(result["manual_wording_and_visible_evidence_review_required"])
        self.assertEqual(result["issue_evidence_audit"][0]["certainty"], "uncertain")
        self.assertEqual(row, original)

    def test_old_zero_record_retains_frozen_protocol_behavior(self):
        row, spec = fixture()
        row["details"].pop("diagnostic_score")
        self.assertFalse(verified_failure_zero(row, spec))
        self.assertIn(FLAG, inspect_record(row, spec)["structural_flags"])

    def test_claimed_zero_and_training_flags_do_not_replace_reducer_checks(self):
        row, spec = fixture()
        row["details"]["scoring"] = {
            "total_score": 0, "review_required": False, "review_reasons": [],
            "independent_failure_confirmed": True,
        }
        self.assertFalse(verified_failure_zero(row, spec))
        self.assertIn(FLAG, inspect_record(row, spec)["structural_flags"])

    def test_corrupted_provenance_never_earns_exception(self):
        mutations = {
            "report hash": lambda r: r["details"]["requirement_scope_audits"][0].update(report_sha256="0" * 64),
            "task source": lambda r: r["details"]["task_contract"].update(source_text="Another task."),
            "row contract hash": lambda r: r.update(task_contract_sha256="0" * 64),
            "unsupplied frame": lambda r: r["details"]["evidence_report"]["observations"][0].update(frames=[0, 2]),
            "wrong timestamp": lambda r: r["details"]["frame_manifest"][1].update(timestamp_seconds=5),
            "diagnostic score": lambda r: r["details"].update(diagnostic_score=0.99),
            "task confidence": lambda r: r["details"]["evidence_report"]["task_assessment"].update(confidence="low"),
            "no failed requirement": lambda r: r["details"]["evidence_report"]["requirement_checks"][0].update(status="uncertain"),
            "unknown resolution": lambda r: r["details"].update(failure_reward_resolution={"confirmed": True}),
            "input mismatch": lambda r: r.update(prompt="Another task."),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                row, spec = fixture()
                mutation(row)
                self.assertFalse(verified_failure_zero(row, spec))
                self.assertIn(FLAG, inspect_record(row, spec)["structural_flags"])

    def test_unknown_component_or_nonfailed_task_cannot_use_exception(self):
        for verdict in ("partial", "complete", "unobservable"):
            with self.subTest(verdict=verdict):
                row, spec = fixture()
                row["details"]["evidence_report"]["task_assessment"]["verdict"] = verdict
                self.assertFalse(verified_failure_zero(row, spec))
                self.assertIn(FLAG, inspect_record(row, spec)["structural_flags"])

    def test_legacy_review_still_avoids_training_inconsistency_flag(self):
        row, spec = fixture()
        row["details"].pop("diagnostic_score")
        row.update(status="needs_review", new_score=0.3, training_eligible=False)
        row["details"].update(review_required=True, training_eligible=False)
        result = inspect_record(row, spec)
        self.assertFalse(result["training_eligible"])
        self.assertTrue(result["review_required"])
        self.assertNotIn(FLAG, result["structural_flags"])

    def test_target_ambiguity_requires_a_recorded_resolution_not_just_zero(self):
        row, spec = fixture()
        details = row["details"]
        report = details["evidence_report"]
        report["task_assessment"]["target_match"] = "uncertain"
        details["pre_scope_repair_report"] = copy.deepcopy(report)
        details["requirement_scope_audits"][0]["report_sha256"] = hashlib.sha256(json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        resolution = {
            "schema_version": RESOLUTION_VERSION, "decision": "failure_established",
            "requirement_id": "R1", "evidence": ["E1"], "confidence": "high",
            "independent_of_unresolved": True,
            "reason": "No block enters the basket; exact block identity cannot change that.",
            "uncertainty_analysis": "The identity uncertainty does not affect the visible empty basket.",
        }
        diagnostic = score_report(core_report(report))
        diagnostic["scoring_version"] = "evidence-soft-physics-v2-with-scope-review-v5-provisional"
        scoring = apply_training_reward(
            diagnostic, report, details["task_contract"], details["requirement_scope_audits"],
            failure_resolution=resolution,
        )
        details.update(scoring=scoring, failure_reward_resolution=resolution,
                       **{key: scoring[key] for key in (
                           "diagnostic_score", "diagnostic_review_required", "diagnostic_review_reasons",
                       )})
        self.assertFalse(verified_failure_zero(row, spec))
        self.assertIn(FLAG, inspect_record(row, spec)["structural_flags"])
        details["trace"] = [{"stage": "failure_reward_resolution", "output": copy.deepcopy(resolution)}]
        self.assertTrue(verified_failure_zero(row, spec))
        self.assertNotIn(FLAG, inspect_record(row, spec)["structural_flags"])
        details["trace"][0]["output"]["reason"] = "Different resolution."
        self.assertFalse(verified_failure_zero(row, spec))

    def test_exception_does_not_relax_black_or_motion_control_behavior(self):
        for sample_id in (BLACK, JUMP):
            with self.subTest(sample_id=sample_id):
                row, spec = fixture()
                row["sample_id"] = spec["sample_id"] = sample_id
                result = inspect_record(row, spec)
                self.assertNotIn(FLAG, result["structural_flags"])
                self.assertFalse(result["behavior_check_pass"])


if __name__ == "__main__":
    unittest.main()
