"""Tests for real failure modes found in the legacy reward reports."""

import copy
import unittest

from reward_as_agent.evidence_schema import (
    EvidenceValidationError,
    score_report,
    validate_report,
)


def report_fixture():
    return {
        "schema_version": "evidence-v2",
        "task": {
            "requested_action": "Put the red block into the basket.",
            "target_description": "Red block and basket.",
            "final_state_requirement": "Block resting inside the basket after gripper release.",
        },
        "observations": [
            {"id": "E1", "frames": [0, 17], "description": "The gripper lifts the red block."},
            {"id": "E2", "frames": [53, 76], "description": "The block stays in the basket after release."},
        ],
        "task_assessment": {
            "verdict": "complete", "confidence": "high", "evidence": ["E1", "E2"],
            "reason": "The requested block is released into the basket.", "target_match": "match",
        },
        "physics_assessment": {
            "verdict": "plausible", "confidence": "high", "evidence": ["E1", "E2"],
            "issues": [], "reason": "Block transport, release and support are consistent.",
        },
        "visual_assessment": {
            "verdict": "clear", "confidence": "high", "evidence": ["E1", "E2"],
            "reason": "Relevant objects and final state are identifiable.",
        },
        "uncertainties": [],
    }


def add_issue(report, severity="major", certainty="confirmed"):
    report["physics_assessment"]["issues"] = [{
        "kind": "interpenetration", "severity": severity, "certainty": certainty,
        "evidence": ["E1"], "reason": "The object visibly passes through the closed gripper.",
        "alternative_explanation": "Ordinary finger occlusion cannot explain the visible exit path.",
    }]
    if certainty == "confirmed":
        report["physics_assessment"]["verdict"] = f"{severity}_defect"
    return report


class EvidenceSchemaTests(unittest.TestCase):
    def test_observation_must_refer_to_actual_input_not_prompt_frame_count(self):
        report = report_fixture()
        report["observations"][1]["frames"] = [53, 999]
        with self.assertRaisesRegex(EvidenceValidationError, "unsupplied source frame IDs.*999"):
            validate_report(report, [0, 17, 53, 76])

    def test_dangling_evidence_reference_is_not_accepted(self):
        report = report_fixture()
        report["task_assessment"]["evidence"] = ["E43"]
        with self.assertRaisesRegex(EvidenceValidationError, "unknown evidence ID 'E43'"):
            validate_report(report, [0, 17, 53, 76])

    def test_two_crops_of_same_source_frame_cannot_confirm_physics_defect(self):
        report = add_issue(report_fixture())
        report["observations"][0]["frames"] = [17]
        report["observations"].append({"id": "E3", "frames": [17], "description": "Detail crop of frame 17."})
        report["physics_assessment"]["issues"][0]["evidence"] = ["E1", "E3"]
        with self.assertRaisesRegex(EvidenceValidationError, "two distinct source frames"):
            validate_report(report, [0, 17, 53, 76])

    def test_confirmed_issue_cannot_receive_plausible_verdict(self):
        report = add_issue(report_fixture())
        report["physics_assessment"]["verdict"] = "plausible"
        with self.assertRaisesRegex(EvidenceValidationError, "incompatible with confirmed"):
            validate_report(report, [0, 17, 53, 76])

    def test_major_verdict_cannot_be_inferred_from_uncertain_or_minor_issue(self):
        for severity, certainty in (("major", "uncertain"), ("minor", "confirmed")):
            with self.subTest(severity=severity, certainty=certainty):
                report = add_issue(report_fixture(), severity, certainty)
                report["physics_assessment"]["verdict"] = "major_defect"
                with self.assertRaisesRegex(EvidenceValidationError, "confirmed major issue"):
                    validate_report(report, [0, 17, 53, 76])

    def test_wrong_target_cannot_be_declared_complete(self):
        for verdict in ("complete", "mostly_complete"):
            with self.subTest(verdict=verdict):
                report = report_fixture()
                report["task_assessment"].update(verdict=verdict, target_match="mismatch")
                with self.assertRaises(EvidenceValidationError):
                    validate_report(report, [0, 17, 53, 76])

    def test_uncertain_target_cannot_be_declared_complete(self):
        report = report_fixture()
        report["task_assessment"]["target_match"] = "uncertain"
        with self.assertRaisesRegex(EvidenceValidationError, "complete requires"):
            validate_report(report, [0, 17, 53, 76])

    def test_missing_physics_evidence_cannot_establish_plausibility(self):
        report = report_fixture()
        report["physics_assessment"]["evidence"] = []
        with self.assertRaisesRegex(EvidenceValidationError, "observable claims require"):
            validate_report(report, [0, 17, 53, 76])

    def test_no_model_numeric_polarity_field_is_accepted(self):
        report = report_fixture()
        report["physics_assessment"]["score"] = 1
        with self.assertRaisesRegex(EvidenceValidationError, "numeric model scores are not accepted"):
            validate_report(report, [0, 17, 53, 76])

    def test_boolean_frame_id_is_not_accepted_as_integer(self):
        report = report_fixture()
        report["observations"][0]["frames"] = [True]
        with self.assertRaisesRegex(EvidenceValidationError, "integer source frame ID"):
            validate_report(report, [0, 17, 53, 76])

    def test_duplicate_evidence_and_wrong_json_types_fail_with_paths(self):
        mutations = [
            (lambda r: r["observations"][1].update(id="E1"), "observations\\[1\\].id"),
            (lambda r: r["task_assessment"].update(evidence="E1"), "task_assessment.evidence"),
            (lambda r: r["visual_assessment"].pop("reason"), "visual_assessment"),
            (lambda r: r.update(uncertainties=[None]), "uncertainties\\[0\\]"),
        ]
        for mutate, path in mutations:
            report = report_fixture()
            mutate(report)
            with self.subTest(path=path), self.assertRaisesRegex(EvidenceValidationError, path):
                validate_report(report, [0, 17, 53, 76])

    def test_reducer_is_monotonic_in_task_even_with_major_physics_defect(self):
        for physical_verdict in ("plausible", "minor_defect", "major_defect"):
            values = []
            for task_verdict in ("failed", "partial", "mostly_complete", "complete"):
                report = report_fixture()
                if physical_verdict != "plausible":
                    add_issue(report, physical_verdict.removesuffix("_defect"))
                report["task_assessment"]["verdict"] = task_verdict
                values.append(score_report(report)["total_score"])
            with self.subTest(physics=physical_verdict):
                self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
                self.assertTrue(all(0 <= value <= 1 for value in values))

    def test_physics_penalty_is_meaningful_without_destroying_task_signal(self):
        clear = score_report(report_fixture())["total_score"]
        minor = score_report(add_issue(report_fixture(), "minor"))["total_score"]
        major = score_report(add_issue(report_fixture()))["total_score"]
        self.assertGreater(clear, minor)
        self.assertGreater(minor, major)
        self.assertGreater(clear - major, 0.3)
        self.assertGreater(major, 0.3)

    def test_visual_degradation_reduces_reward_but_cannot_erase_task_progress(self):
        report = report_fixture()
        values = []
        for verdict in ("severe_degradation", "minor_degradation", "clear"):
            report["visual_assessment"]["verdict"] = verdict
            values.append(score_report(report)["total_score"])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertGreater(values[0], 0.5)

    def test_uncertain_physics_issue_marks_review_without_confirmed_penalty(self):
        baseline = score_report(report_fixture())
        report = add_issue(report_fixture(), certainty="uncertain")
        result = score_report(report)
        self.assertEqual(result["total_score"], baseline["total_score"])
        self.assertTrue(result["review_required"])
        self.assertFalse(baseline["review_required"])

    def test_unobservable_components_abstain_instead_of_zero_reward(self):
        for component in ("task_assessment", "physics_assessment", "visual_assessment"):
            with self.subTest(component=component):
                report = report_fixture()
                report[component].update(verdict="unobservable", confidence="low", evidence=[])
                result = score_report(report)
                self.assertIsNone(result["total_score"])
                self.assertTrue(result["review_required"])
                self.assertTrue(any("unobservable" in reason for reason in result["review_reasons"]))

    def test_low_confidence_is_reviewed_not_silently_reinterpreted_as_failure(self):
        report = report_fixture()
        report["task_assessment"]["confidence"] = "low"
        result = score_report(report)
        self.assertEqual(result["total_score"], 1.0)
        self.assertTrue(result["review_required"])

    def test_generic_visibility_notes_do_not_gate_an_observable_judgement(self):
        report = report_fixture()
        report["uncertainties"] = [
            "Intermediate frames between sampled images are not supplied.",
            "Images do not establish microscopic material properties irrelevant to this task.",
        ]
        for key in ("task_assessment", "physics_assessment", "visual_assessment"):
            report[key]["confidence"] = "medium"
        before = copy.deepcopy(report)
        result = score_report(report)
        self.assertEqual(result["total_score"], 1.0)
        self.assertFalse(result["review_required"])
        self.assertEqual(result["review_reasons"], [])
        self.assertEqual(report, before)
        self.assertEqual(result["scoring_version"], "evidence-soft-physics-v2-provisional")

    def test_material_visibility_limitation_mapped_to_low_confidence_still_requires_review(self):
        report = report_fixture()
        report["uncertainties"] = ["The gripper obscures whether the block was actually released."]
        report["task_assessment"].update(verdict="mostly_complete", confidence="low")
        result = score_report(report)
        self.assertIsNotNone(result["total_score"])
        self.assertTrue(result["review_required"])
        self.assertIn("task: low confidence", result["review_reasons"])

    def test_wrong_target_partial_progress_gets_less_than_correct_target_partial(self):
        report = report_fixture()
        report["task_assessment"]["verdict"] = "partial"
        matched = score_report(report)["total_score"]
        report["task_assessment"]["target_match"] = "mismatch"
        self.assertLess(score_report(report)["total_score"], matched)

    def test_validator_does_not_mutate_report_or_guess_polarity_from_keywords(self):
        report = report_fixture()
        report["task_assessment"].update(verdict="failed", reason="Perfect-looking texture alone does not complete the task.")
        before = copy.deepcopy(report)
        validate_report(report, [0, 17, 53, 76])
        result = score_report(report)
        self.assertEqual(result["component_scores"]["task"], 0.0)
        self.assertEqual(before, report)


if __name__ == "__main__":
    unittest.main()
