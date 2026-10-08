import copy
import unittest
from test_requirement_audit import fixture
from reward_as_agent.focused_audit import focused_prompt, validate_focused_audit


class FocusedAuditTests(unittest.TestCase):
    def test_own_quote_and_exact_target_remain_required(self):
        contract, report, full = fixture()
        audit = {'schema_version': full['schema_version'], 'checks': full['checks'][:1]}
        original = copy.deepcopy((contract, report, audit))
        validate_focused_audit(audit, contract, report, 'R1')
        self.assertEqual((contract, report, audit), original)
        for target in ('R2', 'R99'):
            with self.assertRaises(ValueError):
                validate_focused_audit(audit, contract, report, target)
        audit['checks'][0]['issues'][0]['claim_quote'] = report['requirement_checks'][1]['reason']
        with self.assertRaises(ValueError):
            validate_focused_audit(audit, contract, report, 'R1')

    def test_no_unaudited_targets_can_be_smuggled_into_result(self):
        contract, report, full = fixture()
        with self.assertRaises(ValueError):
            validate_focused_audit(full, contract, report, 'R1')

    def test_context_retains_source_and_other_checks_without_score(self):
        contract, report, _ = fixture()
        report['total_score'] = 'EXCLUDED_SCORE'
        prompt = focused_prompt(contract, report, 'R1')
        self.assertIn(contract['source_text'], prompt)
        self.assertIn(report['requirement_checks'][1]['reason'], prompt)
        self.assertNotIn('EXCLUDED_SCORE', prompt)
        self.assertTrue(prompt.startswith('本次唯一审核目标是 R1'))
        self.assertIn('"requirement_id": "R1", "issues": []', prompt)


class FocusedAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_retains_independent_findings(self):
        from reward_as_agent.evidence_pipeline import EvidencePipeline
        contract, report, full = fixture()
        pipeline = EvidencePipeline.__new__(EvidencePipeline)
        pipeline.audit_mode = 'hybrid'
        async def stage(name, *args):
            if name.endswith('_joint'):
                return copy.deepcopy(full)
            target = name.rsplit('_',1)[1]
            return {'schema_version':full['schema_version'],
                    'checks':[{'requirement_id':target,'issues':[]}]}
        pipeline.stage = stage
        result = await pipeline.audit_requirements(report,contract,'audit',[])
        self.assertEqual(result, full)

    async def test_all_targets_are_actually_checked_and_order_is_preserved(self):
        from reward_as_agent.evidence_pipeline import EvidencePipeline
        contract, report, _ = fixture()
        pipeline = EvidencePipeline.__new__(EvidencePipeline)
        pipeline.audit_mode = 'focused'
        called = []
        async def stage(name, prompt, payload, frames, ids, validator, trace):
            target = name.rsplit('_', 1)[1]
            called.append(target)
            result = {'schema_version':'requirement-scope-audit-v1',
                      'checks':[{'requirement_id':target,'issues':[]}]}
            validator(result, set())
            return result
        pipeline.stage = stage
        result = await pipeline.audit_requirements(report, contract, 'audit', [])
        self.assertCountEqual(called, ['R1', 'R2'])
        self.assertEqual([x['requirement_id'] for x in result['checks']], ['R1', 'R2'])

    async def test_failed_target_cannot_be_replaced_by_empty_issues(self):
        from reward_as_agent.evidence_pipeline import EvidencePipeline
        contract, report, _ = fixture()
        pipeline = EvidencePipeline.__new__(EvidencePipeline)
        pipeline.audit_mode = 'focused'
        async def stage(name, *args):
            if name.endswith('R2'):
                raise ValueError('Target audit failed')
            return {'schema_version':'requirement-scope-audit-v1',
                    'checks':[{'requirement_id':'R1','issues':[]}]}
        pipeline.stage = stage
        with self.assertRaisesRegex(ValueError, 'Target audit failed'):
            await pipeline.audit_requirements(report, contract, 'audit', [])


if __name__ == '__main__':
    unittest.main()
