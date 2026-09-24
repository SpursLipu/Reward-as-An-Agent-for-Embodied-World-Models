"""Offline request/cache/trace checks; normal service dependencies are required."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from reward_as_agent.config import get_settings
from reward_as_agent import doubao
from reward_as_agent.evidence_pipeline import EvidencePipeline, MODEL_CONTEXT_VERSION, PIPELINE_VERSION


MESSAGES = [{'role': 'user', 'content': [
    {'type': 'text', 'text': '中文'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AA=='}},
]}]


def settings(temperature=None):
    return SimpleNamespace(model='doubao-seed-2-1-pro-260628', temperature=temperature,
                           max_tokens=128, api_base='https://example.invalid/api/v3',
                           max_retries=0, llm_timeout=10)


def completed(**extra):
    return {'id': 'offline-response', 'status': 'completed', 'usage': {},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ready":true}'}]}], **extra}


class SamplingHelpersTests(unittest.TestCase):
    def test_payload_omission_zero_and_order(self):
        absent = doubao.build_responses_payload(MESSAGES, settings())
        zero = doubao.build_responses_payload(MESSAGES, settings(0.0))
        self.assertNotIn('temperature', absent)
        self.assertEqual(zero.pop('temperature'), 0.0)
        self.assertEqual(zero, absent)
        self.assertNotIn('top_p', absent)
        self.assertEqual(absent['input'][0]['content'][0], {'type': 'input_text', 'text': '中文'})
        self.assertEqual(absent['input'][0]['content'][1]['type'], 'input_image')
        for invalid in (True, '0', float('nan'), float('inf'), -0.1, 2.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                doubao.build_responses_payload(MESSAGES, settings(invalid))

    def test_echo_absent_null_zero_are_distinct(self):
        self.assertEqual(doubao.sampling_fields({})['temperature'], {'present': False, 'value': None})
        self.assertEqual(doubao.sampling_fields({'temperature': None})['temperature'], {'present': True, 'value': None})
        self.assertEqual(doubao.sampling_fields({'temperature': 0})['temperature'], {'present': True, 'value': 0})
        self.assertNotIn('secret', json.dumps(doubao.sampling_fields({'temperature': 'secret'})))

    def test_incompatible_cache_cannot_supply_provenance(self):
        payload = doubao.build_responses_payload(MESSAGES, settings(0))
        expected = doubao.request_metadata(payload, doubao.encode_request_payload(payload))
        for cached, status in (({}, 'missing_cached_metadata'),
                               ({**expected, 'request_payload_sha256': 'wrong', 'response_sampling': {}}, 'invalid_cached_metadata')):
            result = doubao.cached_sampling_metadata(cached, expected)
            self.assertEqual(result['sampling_metadata_status'], status)
            self.assertIsNone(result['request_payload_sha256'])
            self.assertIsNone(result['request_sampling'])
            self.assertIsNone(result['response_sampling'])


class SamplingTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'ARK_API_KEY': 'offline-secret', 'ARK_RESPONSES_URL': '',
            'REWARD_AS_AGENT_CACHE_DIR': '', 'REWARD_AS_AGENT_CACHE_BYPASS': '',
            'REWARD_TASK_CONTRACT_REGISTRY': '', 'REWARD_EVIDENCE_FRAMES': '8'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def fake_client(self, body, status=200):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = SimpleNamespace(status_code=status, is_error=status >= 400,
                                                    headers={}, json=lambda: body)
        return client

    async def test_actual_wire_body_hash_and_cached_metadata_match(self):
        client = self.fake_client(completed(temperature=0))
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'REWARD_AS_AGENT_CACHE_DIR': directory}), \
                patch.object(doubao.httpx, 'AsyncClient', return_value=client):
            fresh = await doubao.call_doubao(MESSAGES, settings(0))
            cached = await doubao.call_doubao(MESSAGES, settings(0))
            self.assertEqual(client.post.await_count, 1)
            kwargs = client.post.call_args.kwargs
            wire = kwargs['content']
            self.assertNotIn('json', kwargs)
            self.assertEqual(fresh['request_payload_sha256'], hashlib.sha256(wire).hexdigest())
            self.assertEqual(json.loads(wire)['temperature'], 0)
            self.assertNotIn('top_p', json.loads(wire))
            self.assertFalse(fresh['cache_reused'])
            self.assertTrue(cached['cache_reused'])
            for key in ('request_payload_sha256', 'request_sampling', 'response_sampling', 'sampling_metadata_status'):
                self.assertEqual(fresh[key], cached[key])
            self.assertEqual(fresh['response_sampling']['temperature'], {'present': True, 'value': 0})
            self.assertNotIn('offline-secret', Path(fresh['_cache_path']).read_text())
            self.assertNotIn('Authorization', Path(fresh['_cache_path']).read_text())

    async def test_legacy_cache_keeps_metadata_unknown_without_network(self):
        config = settings()
        payload = doubao.build_responses_payload(MESSAGES, config)
        url = config.api_base + '/responses'
        digest = hashlib.sha256(json.dumps({'url': url, 'payload': payload}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'REWARD_AS_AGENT_CACHE_DIR': directory}), \
                patch.object(doubao.httpx, 'AsyncClient') as client:
            (Path(directory) / (digest + '.json')).write_text(json.dumps({'choices': []}))
            result = await doubao.call_doubao(MESSAGES, config)
            client.assert_not_called()
            self.assertTrue(result['cache_reused'])
            self.assertEqual(result['sampling_metadata_status'], 'missing_cached_metadata')
            self.assertIsNone(result['request_payload_sha256'])

    async def test_api_error_retains_request_hash_and_redacts_key(self):
        client = self.fake_client({'error': {'code': 'bad', 'message': 'offline-secret rejected'}}, status=400)
        with patch.object(doubao.httpx, 'AsyncClient', return_value=client):
            with self.assertRaises(doubao.DoubaoRequestError) as caught:
                await doubao.call_doubao(MESSAGES, settings(0))
        self.assertNotIn('offline-secret', str(caught.exception))
        metadata = caught.exception.sampling_metadata
        self.assertEqual(metadata['request_payload_sha256'], hashlib.sha256(client.post.call_args.kwargs['content']).hexdigest())
        self.assertIsNone(metadata['response_sampling'])

    async def test_pipeline_trace_and_fingerprint(self):
        config = replace(get_settings(), provider='doubao', temperature=None)
        omitted = EvidencePipeline(config)
        explicit = EvidencePipeline(replace(config, temperature=0))
        self.assertNotEqual(omitted.evaluator_version, explicit.evaluator_version)
        self.assertEqual(PIPELINE_VERSION, 'evidence-v12-cotracker3-tool-probe')
        with patch.dict(os.environ, {'REWARD_REQUIREMENT_AUDIT_MODE': 'focused'}):
            focused = EvidencePipeline(config)
        self.assertNotEqual(omitted.evaluator_version, focused.evaluator_version)
        self.assertEqual(MODEL_CONTEXT_VERSION, 'evidence-v5.0-dev1')
        payload = doubao.build_responses_payload(MESSAGES, settings(0))
        metadata = {**doubao.request_metadata(payload, doubao.encode_request_payload(payload)),
                    'response_sampling': doubao.sampling_fields({}), 'sampling_metadata_status': 'recorded'}
        reply = {'choices': [{'message': {'content': '{"ready":true}'}}], 'cache_reused': True, **metadata}
        trace = []
        with patch('reward_as_agent.evidence_pipeline.call_llm', new=AsyncMock(return_value=reply)):
            await explicit.stage('offline', 'prompt', {}, [], [], lambda value, frames: None, trace)
        for key, value in metadata.items():
            self.assertEqual(trace[0][key], value)
        self.assertTrue(trace[0]['cache_reused'])
        failed_trace = []
        with patch('reward_as_agent.evidence_pipeline.call_llm', new=AsyncMock(side_effect=doubao.DoubaoRequestError('offline', metadata))):
            with self.assertRaises(doubao.DoubaoRequestError):
                await explicit.stage('offline', 'prompt', {}, [], [], lambda value, frames: None, failed_trace)
        self.assertEqual(failed_trace[0]['request_payload_sha256'], metadata['request_payload_sha256'])


if __name__ == '__main__':
    unittest.main()
