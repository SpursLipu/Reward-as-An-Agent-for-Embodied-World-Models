"""Offline tests for frozen task meaning boundaries and requirement provenance."""
import copy
import unittest

from reward_as_agent.task_contract import (
    TaskContractError, coverage_audit_prompt, extraction_prompt, freeze_contract,
    validate_extraction, validate_requirement_checks, validate_task_contract,
)


SOURCE = "Move the cup slowly, then return the arm to its initial position."


def extraction():
    return {
        "schema_version": "task-contract-v1",
        "task": {"requested_action": "Move the cup slowly and return the arm.",
                 "target_description": "The cup and the operating arm.",
                 "final_state_requirement": "The arm is back at its initial position."},
        "requirements": [
            {"id": "R1", "source_quote": "Move the cup", "text": "Move the cup."},
            {"id": "R2", "source_quote": "slowly", "text": "Perform the cup movement slowly."},
            {"id": "R3", "source_quote": "then return the arm to its initial position",
             "text": "After the cup movement, return the arm to its initial position."},
        ],
        "coverage_notes": [],
    }


def checks():
    return [{"requirement_id": f"R{i}", "status": "met", "evidence": ["E1"],
             "reason": "The cited visible action and final position support this requirement."}
            for i in range(1, 4)]


def report(verdict="complete"):
    return {"task": extraction()["task"], "observations": [{"id": "E1", "frames": [0, 80],
            "description": "The cup moves and the arm returns."}], "task_assessment": {"verdict": verdict}}


class TaskContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = freeze_contract(SOURCE, extraction())

    def test_quote_must_be_exact_contiguous_source_not_semantic_paraphrase(self):
        for quote in ("Move the cup then return", "back to its starting position", "Slowly"):
            with self.subTest(quote=quote):
                candidate = extraction()
                candidate["requirements"][0]["source_quote"] = quote
                with self.assertRaisesRegex(TaskContractError, "exact contiguous substring"):
                    validate_extraction(candidate, SOURCE)

    def test_frozen_source_task_and_requirements_are_fingerprinted_and_copied(self):
        candidate = extraction()
        frozen = freeze_contract(SOURCE, candidate)
        candidate["requirements"][2]["text"] = "Return anywhere."
        self.assertEqual(frozen["requirements"][2]["text"], extraction()["requirements"][2]["text"])
        self.assertEqual(frozen["source_text"], SOURCE)
        validate_task_contract(frozen, SOURCE)
        for mutate in (
            lambda value: value["task"].update(final_state_requirement="Any hovering position."),
            lambda value: value["requirements"][2].update(text="Return anywhere."),
            lambda value: value.update(source_text=SOURCE + " "),
        ):
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(frozen)
                mutate(changed)
                with self.assertRaises(TaskContractError):
                    validate_task_contract(changed)

    def test_source_changed_by_caller_cannot_reuse_contract(self):
        with self.assertRaisesRegex(TaskContractError, "original task supplied"):
            validate_task_contract(self.contract, SOURCE.replace("slowly", "quickly"))

    def test_extraction_rejects_invalid_ids_empty_text_and_extra_numeric_scores(self):
        mutations = (
            lambda value: value["requirements"][1].update(id="R1"),
            lambda value: value["requirements"][0].update(text=""),
            lambda value: value["task"].update(score=1),
            lambda value: value.update(requirements=[]),
        )
        for mutate in mutations:
            candidate = extraction()
            mutate(candidate)
            with self.subTest(mutation=mutate), self.assertRaises(TaskContractError):
                validate_extraction(candidate, SOURCE)

    def test_cannot_drop_add_or_duplicate_a_frozen_requirement(self):
        variants = [checks()[:-1], checks() + [{**checks()[0], "requirement_id": "R4"}],
                    checks() + [checks()[0]]]
        for values in variants:
            with self.subTest(values=values), self.assertRaises(TaskContractError):
                validate_requirement_checks(values, self.contract, report())

    def test_check_cannot_rewrite_requirement_text(self):
        values = checks()
        values[2]["text"] = "Arm need only stop somewhere."
        with self.assertRaisesRegex(TaskContractError, "unknown fields"):
            validate_requirement_checks(values, self.contract, report())

    def test_observable_check_requires_existing_evidence_and_reason(self):
        for status in ("met", "partial", "not_met"):
            values = checks()
            values[1].update(status=status, evidence=[])
            verdict = "complete" if status == "met" else "partial"
            with self.subTest(status=status), self.assertRaisesRegex(TaskContractError, "require evidence"):
                validate_requirement_checks(values, self.contract, report(verdict))
        values = checks()
        values[0]["evidence"] = ["E404"]
        with self.assertRaisesRegex(TaskContractError, "unknown observation ID"):
            validate_requirement_checks(values, self.contract, report())
        values = checks()
        values[0]["reason"] = ""
        with self.assertRaisesRegex(TaskContractError, "nonempty string"):
            validate_requirement_checks(values, self.contract, report())

    def test_complete_cannot_hide_unmet_speed_or_initial_position_requirement(self):
        for index in (1, 2):
            for status in ("partial", "not_met", "uncertain", "unobservable"):
                values = checks()
                values[index]["status"] = status
                with self.subTest(index=index, status=status), self.assertRaisesRegex(TaskContractError, "every frozen requirement"):
                    validate_requirement_checks(values, self.contract, report())

    def test_noncomplete_cannot_be_based_on_all_met_requirements(self):
        for verdict in ("mostly_complete", "partial", "failed", "unobservable"):
            with self.subTest(verdict=verdict), self.assertRaisesRegex(TaskContractError, "non-complete has no unmet task basis"):
                validate_requirement_checks(checks(), self.contract, report(verdict))

    def test_uncertain_or_unobservable_requirement_is_reported_for_review(self):
        for status in ("uncertain", "unobservable"):
            values = checks()
            values[1].update(status=status, evidence=[], reason="The movement interval is not visible.")
            summary = validate_requirement_checks(values, self.contract, report("partial"))
            self.assertTrue(summary["review_required"])
            self.assertEqual(summary["unresolved_requirement_ids"], ["R2"])
            self.assertFalse(summary["all_met"])

    def test_physics_defect_does_not_change_all_met_task_requirements(self):
        value = report()
        value["physics_assessment"] = {"verdict": "major_defect"}
        result = validate_requirement_checks(checks(), self.contract, value)
        self.assertTrue(result["all_met"])
        self.assertFalse(result["review_required"])

    def test_extraction_and_coverage_prompts_are_text_only_and_retain_source(self):
        candidate = extraction()
        before = copy.deepcopy(candidate)
        first = extraction_prompt(SOURCE)
        audit = coverage_audit_prompt(SOURCE, candidate)
        self.assertIn(SOURCE, first)
        self.assertIn(SOURCE, audit)
        self.assertIn("candidate_contract", audit)
        self.assertIn("本阶段没有视频", first)
        self.assertIn("限定条件", audit)
        self.assertEqual(candidate, before)

    def test_valid_quotes_do_not_falsely_prove_semantic_coverage(self):
        # A string validator cannot detect that this legitimate quote omits a qualifier.
        # Coverage is reviewed by the separate text-only stage, never claimed by hashes.
        incomplete = extraction()
        incomplete["requirements"] = incomplete["requirements"][:1]
        self.assertIs(validate_extraction(incomplete, SOURCE), incomplete)
        self.assertIn("引文真实存在不证明语义覆盖完整", extraction_prompt(SOURCE))


if __name__ == "__main__":
    unittest.main()
