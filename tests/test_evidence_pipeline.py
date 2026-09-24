"""Integration checks for evidence provenance, abstention and request lifetime.

No model or network is called. These tests require the normal service dependencies.
"""

import asyncio
from dataclasses import replace
import importlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from reward_as_agent.config import get_settings
from reward_as_agent.evidence_pipeline import EvidencePipeline, validate_observations
from reward_as_agent.evidence_grounding import validate_grounded_report
from reward_as_agent.task_contract import freeze_contract
from reward_as_agent.evidence_video import EvidenceVideo, refinement_indices, uniform_indices


def observation_fixture():
    return {
        "observations": [
            {"id": "E1", "frames": [0, 79], "description": "The block moves from the table to the basket."},
        ],
        "uncertainties": [],
    }


def report_fixture():
    return {
        "schema_version": "evidence-v2",
        "task": {
            "requested_action": "Place block in basket.",
            "target_description": "Block and basket.",
            "final_state_requirement": "Block rests inside basket.",
        },
        "observations": observation_fixture()["observations"],
        "task_assessment": {
            "verdict": "complete", "confidence": "high", "evidence": ["E1"],
            "reason": "The visible final state satisfies the requested placement.", "target_match": "match",
        },
        "physics_assessment": {
            "verdict": "plausible", "confidence": "high", "evidence": ["E1"],
            "issues": [], "reason": "The visible transport and resting state are plausible.",
        },
        "visual_assessment": {
            "verdict": "clear", "confidence": "high", "evidence": ["E1"],
            "reason": "Relevant objects and states are identifiable.",
        },
        "uncertainties": [],
        "requirement_checks": [{"requirement_id": "R1", "status": "met", "evidence": ["E1"],
                                 "reason": "The block rests inside the basket."}],
    }


def contract_fixture():
    return freeze_contract('Place block in basket.', {
        'schema_version': 'task-contract-v1', 'task': report_fixture()['task'],
        'requirements': [{'id': 'R1', 'source_quote': 'Place block in basket.', 'text': 'Place block in basket.'}],
        'coverage_notes': [],
    })


def clean_audit():
    return {'schema_version': 'requirement-scope-audit-v1',
            'checks': [{'requirement_id': 'R1', 'issues': []}]}


def response(value, **extra):
    return {"choices": [{"message": {"content": json.dumps(value)}}], **extra}


class FakeVideo:
    """Keep source IDs visible without actual image encoding or model calls."""

    frames = [object() for _ in range(80)]
    fps = 20.0
    width = 704
    height = 512

    def manifest(self, indices):
        return [{"source_frame_index": i, "timestamp_seconds": i / self.fps, "view": "full"}
                for i in indices]

    def content(self, indices):
        return [{"type": "text", "text": f"SOURCE_FRAME {i}"} for i in indices]


class EvidencePipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"REWARD_EVIDENCE_FRAMES": "8", "REWARD_EVIDENCE_TRACE_DIR": "",
                                                   "REWARD_TASK_CONTRACT_REGISTRY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.pipeline = EvidencePipeline(replace(get_settings(), provider="doubao", max_tokens=1024))
        resolver = patch.object(self.pipeline, 'resolve_task_contract', new=AsyncMock(return_value=contract_fixture()))
        resolver.start()
        self.addCleanup(resolver.stop)
        self.validate_report = lambda value, frames: validate_grounded_report(value, frames, contract_fixture())

    async def test_task_is_absent_from_blind_stage_but_present_for_assessment_and_verification(self):
        task_text = "UNIQUE_TASK_SENTINEL: put the red block into the basket"
        captured = []
        outputs = [observation_fixture(), report_fixture(), report_fixture(), clean_audit()]

        async def model(messages, settings):
            captured.append(messages)
            return response(outputs[len(captured) - 1])

        with patch("reward_as_agent.evidence_pipeline.load_evidence_video", return_value=FakeVideo()), \
                patch("reward_as_agent.evidence_pipeline.call_llm", side_effect=model):
            result = await self.pipeline.process_one_video("not-opened.mp4", task_text, 3)
        self.assertEqual(len(captured), 4)
        self.assertNotIn(task_text, json.dumps(captured[0]))
        self.assertIn(task_text, json.dumps(captured[1]))
        self.assertIn(task_text, json.dumps(captured[2]))
        self.assertEqual(result["planning_api_output"]["index"], 3)
        self.assertTrue(result["training_eligible"])
        self.assertEqual(result["video_metadata"]["decoded_frames"], 80)
        self.assertEqual(result["frame_manifest"][-1]["source_frame_index"], 79)
        self.assertEqual([m['source_frame_index'] for m in result['frame_manifest']], list(range(80)))
        self.assertTrue(result['verification_coverage']['all_decoded_frames_provided'])
        for messages, entry in zip(captured, result['trace']):
            expected = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True,
                                                  separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(entry['messages_sha256'], expected)
        self.assertEqual(len({e['messages_sha256'] for e in result['trace']}), 4)

    async def test_disputed_evidence_adds_real_neighbor_frames_for_verification(self):
        draft = report_fixture()
        draft["physics_assessment"]["issues"] = [{
            "kind": "contact", "severity": "major", "certainty": "uncertain", "evidence": ["E1"],
            "reason": "Contact is briefly unclear.", "alternative_explanation": "Normal occlusion may explain it.",
        }]
        captured = []
        outputs = [observation_fixture(), draft, report_fixture(), clean_audit()]

        async def model(messages, settings):
            captured.append(messages)
            return response(outputs[len(captured) - 1])

        with patch("reward_as_agent.evidence_pipeline.load_evidence_video", return_value=FakeVideo()), \
                patch("reward_as_agent.evidence_pipeline.call_llm", side_effect=model):
            result = await self.pipeline.process_one_video("not-opened.mp4", "Place block", 0)
        original = {int(part["text"].split()[-1]) for part in captured[1][1]["content"][1:]}
        refined = {int(part["text"].split()[-1]) for part in captured[2][1]["content"][1:]}
        self.assertTrue(original < refined)
        self.assertTrue({1, 2, 77, 78} <= refined)
        self.assertTrue(all(0 <= frame_id < 80 for frame_id in refined))
        self.assertEqual(refined, {item["source_frame_index"] for item in result["frame_manifest"]})

    async def test_frozen_task_cannot_be_weakened_during_assessment(self):
        changed = report_fixture()
        changed['task']['requested_action'] = 'Move near the basket.'
        model = AsyncMock(side_effect=[response(observation_fixture()), response(changed),
                                       response(report_fixture()), response(report_fixture()), response(clean_audit())])
        with patch('reward_as_agent.evidence_pipeline.load_evidence_video', return_value=FakeVideo()), \
                patch('reward_as_agent.evidence_pipeline.call_llm', model):
            result = await self.pipeline.process_one_video('not-opened.mp4', 'Place block in basket.', 0)
        self.assertEqual(model.await_count, 5)
        self.assertIn('frozen task contract', result['trace'][1]['validation_error'])
        self.assertEqual(result['evidence_report']['task'], contract_fixture()['task'])

    async def test_unresolved_requirement_is_zero_with_visible_reason(self):
        final = report_fixture()
        final['task_assessment']['verdict'] = 'mostly_complete'
        final['requirement_checks'][0]['status'] = 'uncertain'
        model = AsyncMock(side_effect=[response(observation_fixture()), response(report_fixture()), response(final), response(clean_audit())])
        with patch('reward_as_agent.evidence_pipeline.load_evidence_video', return_value=FakeVideo()), \
                patch('reward_as_agent.evidence_pipeline.call_llm', model):
            result = await self.pipeline.process_one_video('not-opened.mp4', 'Place block in basket.', 0)
        self.assertTrue(result['training_eligible'])
        self.assertEqual(result['total_score'], 0)
        self.assertEqual(result['requirement_summary']['unresolved_requirement_ids'], ['R1'])
        self.assertTrue(any('R1' in reason for reason in result['scoring']['video_quality_gate']['reasons']))

    async def test_task_interpretation_review_is_separate_from_visible_uncertainty(self):
        review = {'status': 'requires_review', 'source_sha256': contract_fixture()['source_sha256'],
                  'reason': 'Two readings of the requested completion state remain unresolved.'}
        self.pipeline.contract_reviews[contract_fixture()['contract_sha256']] = review
        model = AsyncMock(side_effect=[response(observation_fixture()), response(report_fixture()), response(report_fixture()), response(clean_audit())])
        with patch('reward_as_agent.evidence_pipeline.load_evidence_video', return_value=FakeVideo()), \
                patch('reward_as_agent.evidence_pipeline.call_llm', model):
            result = await self.pipeline.process_one_video('not-opened.mp4', 'Place block in basket.', 0)
        self.assertTrue(result['requirement_summary']['all_met'])
        self.assertEqual(result['scoring']['total_score'], 1.0)
        self.assertFalse(result['training_eligible'])
        self.assertEqual(result['task_contract_review'], review)
        self.assertTrue(result['scoring']['review_reasons'][0].startswith('task contract:'))

    async def test_contract_preparation_uses_text_only_and_freezes_audited_result(self):
        extraction = {key: contract_fixture()[key] for key in ('schema_version', 'task', 'requirements', 'coverage_notes')}
        model = AsyncMock(side_effect=[response(extraction), response(extraction)])
        trace = []
        with patch('reward_as_agent.evidence_pipeline.call_llm', model):
            contract = await self.pipeline.prepare_task_contract('Place block in basket.', trace)
        self.assertEqual(contract, contract_fixture())
        self.assertEqual([entry['stage'] for entry in trace], ['task_contract_extraction', 'task_contract_coverage_audit'])
        for call in model.await_args_list:
            self.assertNotIn('image_url', json.dumps(call.args[0]))

    async def test_unknown_component_returns_training_zero(self):
        final = report_fixture()
        final["task_assessment"].update(verdict="unobservable", confidence="low", evidence=[])
        final['requirement_checks'][0].update(status='unobservable', evidence=[])
        model = AsyncMock(side_effect=[response(observation_fixture()), response(report_fixture()), response(final), response(clean_audit())])
        with patch("reward_as_agent.evidence_pipeline.load_evidence_video", return_value=FakeVideo()), \
                patch("reward_as_agent.evidence_pipeline.call_llm", model):
            result = await self.pipeline.process_one_video("not-opened.mp4", "Place block", 0)
        self.assertEqual(result["scoring"]["total_score"], 0)
        self.assertEqual(result["total_score"], 0)
        self.assertFalse(result["review_required"])
        self.assertTrue(result["training_eligible"])

    async def test_low_confidence_returns_training_zero_and_preserves_diagnostic(self):
        final = report_fixture()
        final["task_assessment"]["confidence"] = "low"
        model = AsyncMock(side_effect=[response(observation_fixture()), response(report_fixture()), response(final), response(clean_audit())])
        with patch("reward_as_agent.evidence_pipeline.load_evidence_video", return_value=FakeVideo()), \
                patch("reward_as_agent.evidence_pipeline.call_llm", model):
            result = await self.pipeline.process_one_video("not-opened.mp4", "Place block", 0)
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["diagnostic_score"], 1.0)
        self.assertFalse(result["review_required"])
        self.assertTrue(result["training_eligible"])

    async def test_invalid_cached_frame_citation_is_evicted_before_structural_retry(self):
        invalid = report_fixture()
        invalid["observations"][0]["frames"] = [999]
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "invalid.json"
            cache.write_text("cached malformed report")
            calls = []

            async def model(messages, settings):
                calls.append(messages)
                if len(calls) == 1:
                    return response(invalid, _cache_path=str(cache), cache_reused=True)
                self.assertFalse(cache.exists())
                self.assertIn("999", messages[-1]["content"])
                return response(report_fixture())

            trace = []
            with patch("reward_as_agent.evidence_pipeline.call_llm", side_effect=model):
                final = await self.pipeline.stage("assessment", "system", {}, [], [0, 79], self.validate_report, trace)
        self.assertEqual(final, report_fixture())
        self.assertEqual(len(trace), 2)
        self.assertIn("validation_error", trace[0])

    async def test_malformed_cached_transport_shape_is_evicted_and_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "invalid.json"
            cache.write_text("cached malformed transport")
            calls = []

            async def model(messages, settings):
                calls.append(messages)
                if len(calls) == 1:
                    return {"choices": [], "_cache_path": str(cache)}
                self.assertFalse(cache.exists())
                return response(report_fixture())

            with patch("reward_as_agent.evidence_pipeline.call_llm", side_effect=model):
                result = await self.pipeline.stage("assessment", "system", {}, [], [0, 79], self.validate_report, [])
        self.assertEqual(result, report_fixture())
        self.assertEqual(len(calls), 2)

    async def test_stage_validation_retries_are_bounded_and_do_not_synthesize_score(self):
        model = AsyncMock(return_value=response({"score": 0}))
        trace = []
        with patch("reward_as_agent.evidence_pipeline.call_llm", model):
            with self.assertRaisesRegex(ValueError, "bounded retries"):
                await self.pipeline.stage("assessment", "system", {}, [], [0, 79], self.validate_report, trace)
        self.assertEqual(model.await_count, 3)
        self.assertEqual(len([t for t in trace if 'attempt' in t]), 3)
        self.assertIn('Evidence validation exhausted', trace[-1]['error'])
        self.assertTrue(all("validation_error" in entry for entry in trace if 'attempt' in entry))

    async def test_cancelling_model_call_does_not_launch_repair_or_next_stage(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def model(messages, settings):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        with patch("reward_as_agent.evidence_pipeline.load_evidence_video", return_value=FakeVideo()), \
                patch("reward_as_agent.evidence_pipeline.call_llm", side_effect=model) as mocked:
            task = asyncio.create_task(self.pipeline.process_one_video("not-opened.mp4", "Place block", 0))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(cleaned.is_set())


class BlindObservationContractTests(unittest.TestCase):
    def test_blind_ledger_cannot_contain_extra_task_or_score_fields(self):
        for extra in ({"task": "It succeeded"}, {"score": 1.0}):
            with self.subTest(extra=extra):
                value = {**observation_fixture(), **extra}
                with self.assertRaises(ValueError):
                    validate_observations(value, {0, 79})

    def test_blind_ledger_uses_same_frame_and_id_rules_as_assessment(self):
        mutations = [
            lambda value: value["observations"][0].update(id="whatever"),
            lambda value: value["observations"][0].update(frames=[0, 0]),
            lambda value: value["observations"][0].update(score=1),
            lambda value: value.update(uncertainties=[""]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                value = observation_fixture()
                mutate(value)
                with self.assertRaises(ValueError):
                    validate_observations(value, {0, 79})


class EvidenceVideoTests(unittest.TestCase):
    def test_sampling_preserves_first_last_and_refinement_is_bounded(self):
        sampled = uniform_indices(80, 8)
        self.assertEqual((sampled[0], sampled[-1]), (0, 79))
        self.assertEqual(len(sampled), 8)
        report = report_fixture()
        report["task_assessment"]["confidence"] = "low"
        refined = refinement_indices(report, sampled, 80, max_additional=3)
        self.assertTrue(set(sampled) <= set(refined))
        self.assertLessEqual(len(set(refined) - set(sampled)), 3)
        self.assertTrue(all(0 <= value < 80 for value in refined))

    def test_encoding_uses_original_frame_objects_and_high_jpeg_quality(self):
        frames = [object(), object()]

        class Encoder:
            IMWRITE_JPEG_QUALITY = "quality"

            def __init__(self):
                self.calls = []

            def imencode(self, extension, frame, options):
                self.calls.append((extension, frame, options))
                return True, b"jpeg"

        encoder = Encoder()
        video = EvidenceVideo(frames, 20.0, 704, 512)
        with patch("reward_as_agent.evidence_video.get_cv2", return_value=encoder):
            content = video.content([0, 1])
        self.assertIs(encoder.calls[0][1], frames[0])
        self.assertIs(encoder.calls[1][1], frames[1])
        self.assertTrue(all(call[2] == ["quality", 90] for call in encoder.calls))
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "text", "image_url"])
        self.assertIn("SOURCE_FRAME 1", content[2]["text"])
        self.assertIn("timestamp_seconds=0.05", content[2]["text"])


class EvidenceAPIContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_module = importlib.import_module("reward_as_agent.app")

    def test_review_result_cannot_be_mistaken_for_success_by_training_client(self):
        for total in (-1, 1.0):
            with self.subTest(total=total):
                result = self.app_module.response_from_result({
                    "total_score": total, "planning_api_output": {"index": 5},
                    "review_required": True, "scoring": {"review_reasons": ["low confidence"]},
                })
                self.assertEqual(result["status"], "needs_review")
                self.assertIsNone(result["score"])
                self.assertEqual(result["provisional_score"], None if total == -1 else total)

    def test_zero_training_reward_and_diagnostic_review_are_separate(self):
        result = self.app_module.response_from_result({
            'total_score': 0.0, 'planning_api_output': {'index': 0},
            'review_required': False, 'training_eligible': True,
            'scoring': {'review_reasons': []},
            'evidence_report': {'task_assessment': {'verdict': 'failed'}},
            'diagnostic_score': 0.3, 'diagnostic_review_required': True,
            'diagnostic_review_reasons': ['physics: uncertain motion'],
        })
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['task_verdict'], 'failed')
        self.assertTrue(result['training_eligible'])
        self.assertEqual(result['diagnostic_score'], 0.3)
        self.assertTrue(result['diagnostic_review_required'])

    async def test_closing_stream_awaits_inflight_model_cleanup(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def evaluate(pipeline, path, description, index):
            if index == 0:
                await started.wait()
                return {"total_score": 1.0, "planning_api_output": {"index": index}}
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        with patch.object(self.app_module, "process_one_video_safe", side_effect=evaluate):
            stream = self.app_module.iter_evaluation_results(["fast", "slow"], "Place block", None)
            first = json.loads(await stream.__anext__())
            self.assertEqual(first["index"], 0)
            await stream.aclose()
            self.assertTrue(cleaned.is_set(), "Stream closed before cancelled model request cleaned up")


if __name__ == "__main__":
    unittest.main()
