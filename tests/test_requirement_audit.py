"""Offline provenance tests; no test claims to validate model semantic accuracy."""
import copy
import json
import unittest

from reward_as_agent.requirement_audit import (
    AUDIT_VERSION, ISSUE_KINDS, RequirementAuditError,
    audit_prompt, validate_requirement_audit,
)
from reward_as_agent.task_contract import freeze_contract


def fixture():
    source = "The robot carries a parcel, then hands it to the recipient."
    contract = freeze_contract(source, {
        "schema_version": "task-contract-v1",
        "task": {"requested_action": "Carry then hand over a parcel.",
                 "target_description": "A parcel and its recipient.",
                 "final_state_requirement": "The parcel has been handed to its recipient."},
        "requirements": [
            {"id": "R1", "source_quote": "carries a parcel", "text": "Carry the parcel."},
            {"id": "R2", "source_quote": "then hands it to the recipient",
             "text": "After carrying, hand the parcel to its recipient."},
        ],
        "coverage_notes": [],
    })
    report = {
        "task": copy.deepcopy(contract["task"]),
        "requirement_checks": [
            {"requirement_id": "R1", "status": "partial", "evidence": ["E1"],
             "reason": "The parcel was carried, but the robot did not keep holding it after handover."},
            {"requirement_id": "R2", "status": "met", "evidence": ["E1"],
             "reason": "The recipient took the parcel as the robot released it."},
        ],
        "observations": [{"id": "E1", "frames": [0, 1],
                          "description": "The recipient holds the parcel and the robot releases it."}],
        "task_assessment": {"verdict": "partial", "reason": "Carrying was judged partial."},
        "physics_assessment": {"verdict": "plausible", "reason": "Contacts were described as plausible."},
        "visual_assessment": {"verdict": "clear", "reason": "Objects were described as visible."},
        "uncertainties": [],
    }
    audit = {"schema_version": AUDIT_VERSION, "checks": [
        {"requirement_id": "R1", "issues": [{
            "kind": "temporal_scope",
            "claim_quote": "the robot did not keep holding it after handover",
            "reference_requirement_ids": ["R2"],
            "reason": "R1 concerns carrying; it does not extend holding beyond the handover in R2.",
        }]},
        {"requirement_id": "R2", "issues": []},
    ]}
    return contract, report, audit


class RequirementAuditTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.report, self.audit = fixture()

    def test_valid_audit_returns_none_and_does_not_mutate_inputs(self):
        before = copy.deepcopy((self.contract, self.report, self.audit))
        self.assertIsNone(validate_requirement_audit(self.audit, self.contract, self.report))
        self.assertEqual((self.contract, self.report, self.audit), before)

    def test_empty_issues_are_allowed_without_claiming_report_correctness(self):
        # Even this fixture's questionable holding reason may be missed by an LLM.
        # Structural validation must not invent an issue or silently change it.
        self.audit["checks"][0]["issues"] = []
        self.assertIsNone(validate_requirement_audit(self.audit, self.contract, self.report))

    def test_exact_quote_is_required_from_own_reason(self):
        variants = [
            "The robot did not keep holding it after handover",  # changed case
            "robot failed to hold after transfer",  # paraphrase
            "The parcel was carried after handover",  # non-contiguous splice
            self.report["requirement_checks"][1]["reason"],  # another requirement
            self.report["observations"][0]["description"],  # an observation
            "carries a parcel",  # original task, not the assessed reason
        ]
        for quote in variants:
            with self.subTest(quote=quote):
                value = copy.deepcopy(self.audit)
                value["checks"][0]["issues"][0]["claim_quote"] = quote
                with self.assertRaisesRegex(RequirementAuditError, "this requirement check.reason"):
                    validate_requirement_audit(value, self.contract, self.report)

    def test_quotes_are_case_sensitive_unicode_substrings_without_normalization(self):
        self.report["requirement_checks"][0]["reason"] = "搬运结束后仍须握持。"
        self.audit["checks"][0]["issues"][0]["claim_quote"] = "结束后仍须握持"
        validate_requirement_audit(self.audit, self.contract, self.report)
        self.audit["checks"][0]["issues"][0]["claim_quote"] = "结束后 仍须握持"
        with self.assertRaises(RequirementAuditError):
            validate_requirement_audit(self.audit, self.contract, self.report)

    def test_audit_ids_cannot_be_missing_added_duplicated_or_reordered(self):
        variants = [self.audit["checks"][:1], self.audit["checks"] + [self.audit["checks"][0]],
                    [self.audit["checks"][0], self.audit["checks"][0]],
                    list(reversed(self.audit["checks"]))]
        for checks in variants:
            with self.subTest(checks=checks), self.assertRaisesRegex(RequirementAuditError, "frozen order"):
                validate_requirement_audit({"schema_version": AUDIT_VERSION, "checks": checks},
                                          self.contract, self.report)

    def test_reordered_report_checks_are_mapped_by_id_not_array_position(self):
        self.report["requirement_checks"].reverse()
        validate_requirement_audit(self.audit, self.contract, self.report)

    def test_reference_ids_must_exist_but_may_be_empty(self):
        issue = self.audit["checks"][0]["issues"][0]
        issue["reference_requirement_ids"] = []
        validate_requirement_audit(self.audit, self.contract, self.report)
        issue["reference_requirement_ids"] = ["R99"]
        with self.assertRaisesRegex(RequirementAuditError, "not present in the frozen contract"):
            validate_requirement_audit(self.audit, self.contract, self.report)

    def test_strict_keys_version_and_json_types(self):
        mutations = [
            lambda x: x.update(schema_version="future"),
            lambda x: x.update(score=0),
            lambda x: x.pop("checks"),
            lambda x: x.update(checks={}),
            lambda x: x["checks"][0].update(verdict="failed"),
            lambda x: x["checks"][0].update(issues={}),
            lambda x: x["checks"][0]["issues"][0].update(kind=[]),
            lambda x: x["checks"][0]["issues"][0].update(kind="wrong_video_fact"),
            lambda x: x["checks"][0]["issues"][0].update(claim_quote=" "),
            lambda x: x["checks"][0]["issues"][0].update(reference_requirement_ids="R2"),
            lambda x: x["checks"][0]["issues"][0].update(reference_requirement_ids=[2]),
            lambda x: x["checks"][0]["issues"][0].update(reason=""),
            lambda x: x["checks"][0]["issues"][0].update(confidence="high"),
            lambda x: x["checks"][0]["issues"][0].pop("reason"),
        ]
        for mutate in mutations:
            value = copy.deepcopy(self.audit)
            mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(RequirementAuditError):
                validate_requirement_audit(value, self.contract, self.report)

    def test_all_declared_issue_types_are_structurally_supported_not_semantically_proven(self):
        # Changing only kind would not make this allegation semantically correct.
        # The validator deliberately does not claim to establish that fact.
        for kind in ISSUE_KINDS:
            self.audit["checks"][0]["issues"][0]["kind"] = kind
            validate_requirement_audit(self.audit, self.contract, self.report)

    def test_invalid_input_reason_mapping_fails_without_mutation(self):
        variants = [[], [self.report["requirement_checks"][0]] * 2,
                    [{**self.report["requirement_checks"][0], "requirement_id": "R9"},
                     self.report["requirement_checks"][1]],
                    [{**self.report["requirement_checks"][0], "reason": ""},
                     self.report["requirement_checks"][1]]]
        for checks in variants:
            value = {**self.report, "requirement_checks": copy.deepcopy(checks)}
            before = copy.deepcopy(value)
            with self.subTest(checks=checks), self.assertRaises(RequirementAuditError):
                validate_requirement_audit(self.audit, self.contract, value)
            self.assertEqual(value, before)

    def test_frozen_contract_mutation_is_rejected(self):
        self.contract["requirements"][0]["text"] = "Hold forever."
        with self.assertRaisesRegex(ValueError, "frozen contract content changed"):
            validate_requirement_audit(self.audit, self.contract, self.report)

    def test_prompt_contains_relevant_text_and_omits_run_scores_and_paths(self):
        self.report.update(total_score=0.123456789, video_path="EXCLUDED_VIDEO_PATH", run_id="EXCLUDED_RUN")
        before = copy.deepcopy((self.contract, self.report))
        prompt = audit_prompt(self.contract, self.report)
        payload = json.loads(prompt.split("\n待审文本数据：\n", 1)[1])
        self.assertEqual(payload["contract"]["source_text"], self.contract["source_text"])
        self.assertEqual(payload["report"]["requirement_checks"], self.report["requirement_checks"])
        self.assertEqual(payload["report"]["observations"], self.report["observations"])
        self.assertEqual(payload["report"]["task_assessment"], self.report["task_assessment"])
        for excluded in ("0.123456789", "EXCLUDED_VIDEO_PATH", "EXCLUDED_RUN"):
            self.assertNotIn(excluded, prompt)
        self.assertIn("不是经过确认的视觉真值", prompt)
        self.assertIn("没有明确文本依据就输出 issues=[]", prompt)
        self.assertEqual((self.contract, self.report), before)


if __name__ == "__main__":
    unittest.main()
