"""Bounded semantic-audit integration; model fixtures are not accuracy labels."""
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_evidence_pipeline as fixtures
from test_evidence_pipeline import (
    FakeVideo, clean_audit, contract_fixture,
    observation_fixture, report_fixture, response,
)
from scripts.evidence.check_contract_results import check_record


def challenged_report():
    report = report_fixture()
    report['requirement_checks'][0]['reason'] = (
        'The block rests inside the basket and must remain there for ten seconds.'
    )
    return report


def scope_issue():
    audit = clean_audit()
    audit['checks'][0]['issues'] = [{
        'kind': 'added_condition', 'claim_quote': 'must remain there for ten seconds',
        'reference_requirement_ids': [],
        'reason': 'The placement requirement does not prescribe a holding duration.',
    }]
    return audit


def failed_report(target='match'):
    report = report_fixture()
    report['task_assessment'].update(verdict='failed', target_match=target,
                                     reason='The block stays on the table throughout.')
    report['requirement_checks'][0].update(status='not_met', reason='No placement occurs.')
    report['observations'][0]['description'] = 'The block stays on the table throughout.'
    return report


def resolution_fixture(established=True):
    return {
        'schema_version': 'failure-reward-resolution-v1',
        'decision': 'failure_established' if established else 'unresolved',
        'requirement_id': 'R1' if established else None,
        'evidence': ['E1'] if established else [],
        'confidence': 'high' if established else 'low',
        'independent_of_unresolved': established,
        'reason': 'No block is placed into the basket.' if established else 'Placement remains unclear.',
        'uncertainty_analysis': 'Neither possible target moves.' if established else 'Target is occluded.',
    }


class ScopePipelineTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.EvidencePipelineTests.setUp

    async def run_outputs(self, outputs):
        model = AsyncMock(side_effect=[response(value) for value in outputs])
        with patch('reward_as_agent.evidence_pipeline.load_evidence_video', return_value=FakeVideo()), \
                patch('reward_as_agent.evidence_pipeline.call_llm', model):
            result = await self.pipeline.process_one_video('not-opened.mp4', 'Place block in basket.', 0)
        return result, model

    async def test_clean_audit_has_no_images_and_does_not_launch_repair(self):
        result, model = await self.run_outputs([
            observation_fixture(), report_fixture(), report_fixture(), clean_audit(),
        ])
        self.assertEqual(model.await_count, 4)
        self.assertEqual([t['stage'] for t in result['trace']], [
            'blind_observation', 'assessment', 'verification', 'requirement_scope_audit',
        ])
        self.assertNotIn('SOURCE_FRAME', json.dumps(model.await_args_list[-1].args[0]))
        self.assertEqual(result['evidence_report'], report_fixture())
        self.assertTrue(result['training_eligible'])

    async def test_failed_task_with_physics_doubt_is_zero_without_new_model_call(self):
        failed = failed_report()
        failed['physics_assessment']['issues'] = [{
            'kind': 'motion', 'severity': 'major', 'certainty': 'uncertain',
            'evidence': ['E1'], 'reason': 'Arm motion may be discontinuous.',
            'alternative_explanation': 'Sparse frames can explain the motion.',
        }]
        result, model = await self.run_outputs([
            observation_fixture(), failed, failed, clean_audit(),
        ])
        self.assertEqual(model.await_count, 4)
        self.assertEqual(result['total_score'], 0.0)
        self.assertTrue(result['training_eligible'])
        self.assertFalse(result['review_required'])
        self.assertTrue(result['diagnostic_review_required'])
        self.assertEqual(result['diagnostic_score'], 0.3)
        self.assertIsNone(result['failure_reward_resolution'])
        self.assertEqual(result['evidence_report'], failed)

    async def test_target_doubt_requires_image_grounded_resolution(self):
        for established in (True, False):
            with self.subTest(established=established):
                failed = failed_report(target='uncertain')
                resolution = resolution_fixture(established)
                result, model = await self.run_outputs([
                    observation_fixture(), failed, failed, clean_audit(), resolution,
                ])
                self.assertEqual(model.await_count, 5)
                self.assertEqual(result['failure_reward_resolution'], resolution)
                self.assertEqual(result['training_eligible'], established)
                self.assertEqual(result['review_required'], not established)
                self.assertEqual(result['trace'][-1]['stage'], 'failure_reward_resolution')
                self.assertEqual(model.await_args_list[-1].args[0][1]['content'][1:],
                                 model.await_args_list[2].args[0][1]['content'][1:])
                self.assertEqual(result['evidence_report'], failed)
                if established:
                    self.assertEqual(result['total_score'], 0.0)
                source = {'prompt': 'Place block in basket.', 'video_path': 'not-opened.mp4'}
                row = {**source, 'details': result,
                       'task_contract_sha256': contract_fixture()['contract_sha256'],
                       'new_score': result['scoring']['total_score'],
                       'training_eligible': established,
                       'status': 'success' if established else 'needs_review'}
                check_record(row, source, contract_fixture(), None)

    async def test_contract_ambiguity_cannot_be_bypassed_by_failed_verdict(self):
        failed = failed_report()
        self.pipeline.contract_reviews[contract_fixture()['contract_sha256']] = {
            'status': 'requires_review', 'source_sha256': contract_fixture()['source_sha256'],
            'reason': 'Two incompatible task readings remain.',
        }
        result, model = await self.run_outputs([
            observation_fixture(), failed, failed, clean_audit(),
        ])
        self.assertEqual(model.await_count, 4)
        self.assertFalse(result['training_eligible'])
        self.assertTrue(result['review_required'])

    async def test_repair_can_confirm_failure_instead_of_forcing_a_higher_score(self):
        failed = report_fixture()
        failed['task_assessment'].update(verdict='failed', reason='The block remains outside the basket.')
        failed['requirement_checks'][0].update(status='not_met', reason='The block remains on the table.')
        failed['observations'][0]['description'] = 'The block remains on the table.'
        result, model = await self.run_outputs([
            observation_fixture(), report_fixture(), challenged_report(), scope_issue(), failed, clean_audit(),
        ])
        self.assertEqual(model.await_count, 6)
        self.assertEqual(result['evidence_report'], failed)
        self.assertEqual(result['scoring']['total_score'], 0.0)
        self.assertEqual(result['diagnostic_score'], 0.3)
        self.assertTrue(result['training_eligible'])
        self.assertEqual(len(result['requirement_scope_audits']), 2)
        original_frames = model.await_args_list[2].args[0][1]['content'][1:]
        repair_frames = model.await_args_list[4].args[0][1]['content'][1:]
        self.assertEqual(repair_frames, original_frames)

    async def test_remaining_issue_keeps_score_but_excludes_training_and_passes_offline_gate_check(self):
        result, model = await self.run_outputs([
            observation_fixture(), report_fixture(), challenged_report(), scope_issue(),
            challenged_report(), scope_issue(),
        ])
        self.assertEqual(model.await_count, 6)
        self.assertEqual(result['scoring']['total_score'], 1.0)
        self.assertFalse(result['training_eligible'])
        self.assertTrue(any('unresolved scope audit' in reason for reason in result['scoring']['review_reasons']))
        source = {'prompt': 'Place block in basket.', 'video_path': 'not-opened.mp4'}
        row = {**source, 'details': result, 'task_contract_sha256': contract_fixture()['contract_sha256'],
               'new_score': 1.0, 'training_eligible': False, 'status': 'needs_review'}
        check_record(row, source, contract_fixture(), None)
        corrupted = copy.deepcopy(row)
        corrupted['details']['requirement_scope_audits'][-1]['report_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'audited report hash'):
            check_record(corrupted, source, contract_fixture(), None)

    async def test_invalid_audit_cannot_be_treated_as_no_issues(self):
        malformed = scope_issue()
        malformed['checks'][0]['issues'][0]['claim_quote'] = 'invented quote'
        model = AsyncMock(side_effect=[response(value) for value in [
            observation_fixture(), report_fixture(), challenged_report(), malformed, malformed, malformed,
        ]])
        with patch('reward_as_agent.evidence_pipeline.load_evidence_video', return_value=FakeVideo()), \
                patch('reward_as_agent.evidence_pipeline.call_llm', model):
            with self.assertRaisesRegex(ValueError, 'requirement_scope_audit failed'):
                await self.pipeline.process_one_video('not-opened.mp4', 'Place block in basket.', 0)
        self.assertEqual(model.await_count, 6)
