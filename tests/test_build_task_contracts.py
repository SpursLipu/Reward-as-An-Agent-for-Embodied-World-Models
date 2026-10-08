"""Seed/import/resume regression checks with a fake pipeline; no network."""
import argparse
import asyncio
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evidence import build_task_contracts as builder
from reward_as_agent.task_contract import freeze_contract


def contract(source, action=None):
    return freeze_contract(source, {
        'schema_version': 'task-contract-v1',
        'task': {'requested_action': action or source,
                 'target_description': 'Synthetic object', 'final_state_requirement': source},
        'requirements': [{'id': 'R1', 'source_quote': source, 'text': source}],
        'coverage_notes': [],
    })


def source_key(source):
    return hashlib.sha256(source.encode()).hexdigest()


def review_for(value):
    return {'source_sha256': value['source_sha256'], 'status': 'requires_review',
            'reason': 'Synthetic unresolved interpretation; not a human label.',
            'reviewer': 'independent_ai_text_audit', 'human_label': False,
            'audit_file': 'CONTRACT_AUDIT.md'}


class FakePipeline:
    calls = []
    evaluator_version = 'offline-preparation-v1'

    def __init__(self, settings):
        pass

    async def prepare_task_contract(self, source, trace):
        self.calls.append(source)
        return contract(source)


class SeedRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kept = contract('Move the cup slowly.')
        self.unrelated = contract('Place a blue cube in a bin.')
        self.seed = {'schema_version': 'task-contract-registry-v1',
                     'preparation_evaluator_version': 'frozen-v4',
                     'contracts': {c['source_sha256']: c for c in (self.kept, self.unrelated)},
                     'contract_reviews': {c['contract_sha256']: review_for(c) for c in (self.kept, self.unrelated)}}
        self.seed_path = self.root / 'seed.json'
        self.write_seed()
        self.sources = {self.kept['source_sha256']: self.kept['source_text'],
                        source_key('Lift the object.'): 'Lift the object.'}
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps([
            {'sample_id': str(i), 'prompt': text, 'video_path': '/not-read.mp4'}
            for i, text in enumerate([self.kept['source_text'], self.kept['source_text'], 'Lift the object.'])]))
        self.output = self.root / 'output'
        FakePipeline.calls = []

    def write_seed(self):
        self.seed_path.write_text(json.dumps(self.seed, ensure_ascii=False))

    def args(self, seed=True, resume=False):
        return argparse.Namespace(manifest=self.manifest, output=self.output, jobs=2,
                                  resume=resume, seed_registry=self.seed_path if seed else None)

    def run_offline(self, args):
        with patch.dict(os.environ, {}, clear=True), patch.object(
                builder, 'load_runtime', return_value=(FakePipeline, lambda: None, lambda *a: [])), contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(builder.run(args))

    def test_imports_only_matching_source_and_keeps_exact_contract_and_ai_review(self):
        original = self.seed_path.read_bytes()
        imported, reviews, info = builder.load_seed_registry(self.seed_path, self.sources)
        self.assertEqual(imported, {self.kept['source_sha256']: self.kept})
        self.assertEqual(reviews, {self.kept['contract_sha256']: review_for(self.kept)})
        self.assertEqual(info['imported_count'], 1)
        self.assertEqual(info['imported_review_count'], 1)
        self.assertEqual(info['seed_registry_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(info['seed_registry_path'], str(self.seed_path.resolve()))
        self.assertEqual(self.seed_path.read_bytes(), original)

    def test_rejects_source_key_and_contract_hash_mismatch_even_for_unrelated_source(self):
        mutations = (
            lambda r: r['contracts'][self.kept['source_sha256']].update(contract_sha256='bad'),
            lambda r: r['contracts'][self.kept['source_sha256']].update(source_sha256='bad'),
            lambda r: r['contracts'].update({'bad-key': r['contracts'].pop(self.kept['source_sha256'])}),
            lambda r: r['contracts'][self.unrelated['source_sha256']].update(contract_sha256='bad'),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(self.seed)
                mutate(candidate)
                self.seed_path.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    builder.load_seed_registry(self.seed_path, self.sources)

    def test_rejects_same_hash_but_changed_caller_source(self):
        changed = dict(self.sources)
        changed[self.kept['source_sha256']] = 'Move the cup quickly.'
        with self.assertRaises(ValueError):
            builder.load_seed_registry(self.seed_path, changed)

    def test_rejects_bad_review_binding_or_missing_nonhuman_flag(self):
        key = self.kept['contract_sha256']
        for field, value in [('source_sha256', 'bad'), ('status', 'approved'), ('human_label', True),
                             ('human_label', None), ('reason', ''), ('reviewer', '')]:
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(self.seed)
                candidate['contract_reviews'][key][field] = value
                self.seed_path.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    builder.load_seed_registry(self.seed_path, self.sources)

    def test_run_only_extracts_remaining_unique_source_and_preserves_flags(self):
        self.assertEqual(self.run_offline(self.args()), 0)
        self.assertEqual(FakePipeline.calls, ['Lift the object.'])
        result = json.loads((self.output / 'registry.json').read_text())
        self.assertEqual(result['contracts'][self.kept['source_sha256']], self.kept)
        self.assertNotIn(self.unrelated['source_sha256'], result['contracts'])
        self.assertEqual(result['contract_reviews'], {self.kept['contract_sha256']: review_for(self.kept)})
        config = json.loads((self.output / 'preparation_config.json').read_text())
        self.assertEqual(config['seed_registry_sha256'], hashlib.sha256(self.seed_path.read_bytes()).hexdigest())
        self.assertEqual(self.run_offline(self.args(resume=True)), 0)
        self.assertEqual(FakePipeline.calls, ['Lift the object.'])

    def test_resume_rejects_changed_seed_bytes_or_dropped_seed(self):
        self.run_offline(self.args())
        self.seed['preparation_evaluator_version'] = 'different-metadata'
        self.write_seed()
        with self.assertRaisesRegex(ValueError, 'identical configuration'):
            self.run_offline(self.args(resume=True))
        with self.assertRaisesRegex(ValueError, 'identical configuration'):
            self.run_offline(self.args(seed=False, resume=True))
        self.assertEqual(FakePipeline.calls, ['Lift the object.'])

    def test_conflicting_existing_contract_or_registry_is_not_overwritten(self):
        changed = contract(self.kept['source_text'], action='A different frozen action')
        for location in ('file', 'registry'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                dest = output / 'contracts' / (self.kept['source_sha256'] + '.json') if location == 'file' else output / 'registry.json'
                dest.parent.mkdir(parents=True, exist_ok=True)
                value = changed if location == 'file' else {'contracts': {self.kept['source_sha256']: changed}}
                before = json.dumps(value).encode()
                dest.write_bytes(before)
                args = self.args()
                args.output = output
                with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                    self.run_offline(args)
                self.assertEqual(dest.read_bytes(), before)
                self.assertFalse((output / 'preparation_config.json').exists())
        self.assertEqual(FakePipeline.calls, [])

    def test_without_seed_keeps_existing_configuration_and_preparation_behavior(self):
        self.assertEqual(self.run_offline(self.args(seed=False)), 0)
        self.assertCountEqual(FakePipeline.calls, list(self.sources.values()))
        result = json.loads((self.output / 'registry.json').read_text())
        self.assertNotIn('seed_registry_sha256', result)
        self.assertNotIn('seed_import', result)
        self.assertNotIn('contract_reviews', result)
        config = json.loads((self.output / 'preparation_config.json').read_text())
        self.assertEqual(set(config), {'preparation_evaluator_version', 'source_manifest_sha256', 'cache_bypass'})
        self.assertEqual(self.run_offline(self.args(seed=False, resume=True)), 0)
        self.assertEqual(len(FakePipeline.calls), 2)


if __name__ == '__main__':
    unittest.main()
