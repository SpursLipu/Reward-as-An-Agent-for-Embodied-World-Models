import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from reward_as_agent.doubao import (DoubaoRequestError, call_doubao, responses_input,
                                    progress_thinking_stream)
from reward_as_agent.llm import retry_llm_call


class DoubaoTests(unittest.TestCase):
    def test_progress_thinking_stream_parses_completed_sse_and_records_payload(self):
        seen = []
        def respond(request):
            payload = __import__('json').loads(request.content)
            seen.append(payload)
            events = [
                {'type': 'response.output_text.delta', 'delta': '{"score":1}'},
                {'type': 'response.completed', 'response': {'id': 'stream-ok', 'status': 'completed',
                    'usage': {'output_tokens': 12}, 'output': [{'type': 'message', 'content': [
                        {'type': 'output_text', 'text': '{"score":1}'}]}]}},
            ]
            body = ''.join('data: ' + __import__('json').dumps(e) + '\n\n' for e in events)
            return httpx.Response(200, text=body, headers={'content-type': 'text/event-stream'})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        settings = SimpleNamespace(api_base='https://example.invalid/api/v3', model='test',
                                   max_tokens=32, max_retries=0, llm_timeout=10)
        async def invoke():
            token = progress_thinking_stream.set(True)
            try:
                return await call_doubao([{'role': 'user', 'content': 'test'}], settings)
            finally:
                progress_thinking_stream.reset(token)
        with patch.dict(os.environ, {'ARK_API_KEY': 'test-only', 'REWARD_AS_AGENT_CACHE_BYPASS': '1'}), \
                patch('reward_as_agent.doubao.httpx.AsyncClient', return_value=client), \
                patch('reward_as_agent.doubao.paced_dispatch', new_callable=AsyncMock):
            result = asyncio.run(invoke())
        self.assertEqual(result['choices'][0]['message']['content'], '{"score":1}')
        self.assertEqual(seen[0]['thinking'], {'type': 'enabled'})
        self.assertTrue(seen[0]['stream'])
        self.assertEqual(seen[0]['max_output_tokens'], 16384)

    def test_progress_thinking_stream_rejects_incomplete_response(self):
        events = [{'type': 'response.incomplete', 'response': {'status': 'incomplete',
                  'incomplete_details': {'reason': 'max_output_tokens'}}}]
        body = ''.join('data: ' + __import__('json').dumps(e) + '\n\n' for e in events)
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text=body, headers={'content-type': 'text/event-stream'})))
        settings = SimpleNamespace(api_base='https://example.invalid/api/v3', model='test',
                                   max_tokens=32, max_retries=0, llm_timeout=10)
        async def invoke():
            token = progress_thinking_stream.set(True)
            try:
                return await call_doubao([{'role': 'user', 'content': 'test'}], settings)
            finally:
                progress_thinking_stream.reset(token)
        with patch.dict(os.environ, {'ARK_API_KEY': 'test-only', 'REWARD_AS_AGENT_CACHE_BYPASS': '1'}), \
                patch('reward_as_agent.doubao.httpx.AsyncClient', return_value=client), \
                patch('reward_as_agent.doubao.paced_dispatch', new_callable=AsyncMock):
            with self.assertRaisesRegex(DoubaoRequestError, 'streamed response incomplete'):
                asyncio.run(invoke())

    def test_remote_protocol_failure_retries_then_recovers(self):
        requests = []
        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.RemoteProtocolError('connection dropped', request=request)
            return httpx.Response(200, json={'status': 'completed', 'output': [
                {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"score":1}'}]}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        settings = SimpleNamespace(api_base='https://example.invalid/api/v3', model='test',
                                   max_tokens=32, max_retries=1, llm_timeout=10)
        with patch.dict(os.environ, {'ARK_API_KEY': 'test-only', 'REWARD_AS_AGENT_CACHE_BYPASS': '1'}), \
                patch('reward_as_agent.doubao.httpx.AsyncClient', return_value=client), \
                patch('reward_as_agent.doubao.paced_dispatch', new_callable=AsyncMock), \
                patch('reward_as_agent.doubao.asyncio.sleep', new_callable=AsyncMock):
            result = asyncio.run(call_doubao([{'role': 'user', 'content': 'test'}], settings))
        self.assertEqual(len(requests), 2)
        self.assertEqual(result['choices'][0]['message']['content'], '{"score":1}')

    def test_exhausted_read_errors_are_bounded_and_preserve_provenance(self):
        requests = []
        def respond(request):
            requests.append(request)
            raise httpx.ReadError('private transport detail', request=request)
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        settings = SimpleNamespace(api_base='https://example.invalid/api/v3', model='test',
                                   max_tokens=32, max_retries=1, llm_timeout=10)
        with patch.dict(os.environ, {'ARK_API_KEY': 'test-only', 'REWARD_AS_AGENT_CACHE_BYPASS': '1'}), \
                patch('reward_as_agent.doubao.httpx.AsyncClient', return_value=client), \
                patch('reward_as_agent.doubao.paced_dispatch', new_callable=AsyncMock), \
                patch('reward_as_agent.doubao.asyncio.sleep', new_callable=AsyncMock):
            with self.assertRaises(DoubaoRequestError) as raised:
                asyncio.run(call_doubao([{'role': 'user', 'content': 'test'}], settings))
        self.assertEqual(len(requests), 2)
        self.assertEqual(str(raised.exception), 'ReadError')
        self.assertEqual(len(raised.exception.sampling_metadata['request_payload_sha256']), 64)

    def test_successful_identical_stage_is_reused_without_another_request(self):
        seen = []
        def respond(request):
            seen.append(request)
            return httpx.Response(200, json={"status": "completed", "id": "test-response", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"score":1}'}]}]})
        settings = SimpleNamespace(api_base="https://example.invalid/api/v3", model="test", max_tokens=4096, max_retries=0, llm_timeout=10)
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        async def twice():
            a = await call_doubao([{"role": "user", "content": "same"}], settings)
            b = await call_doubao([{"role": "user", "content": "same"}], settings)
            return a, b
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ARK_API_KEY": "test-only", "REWARD_AS_AGENT_CACHE_DIR": folder, "REWARD_AS_AGENT_CACHE_BYPASS": "0"}), patch("reward_as_agent.doubao.httpx.AsyncClient", return_value=client):
            first, second = asyncio.run(twice())
            self.assertEqual(len(seen), 1)
            self.assertTrue(second["cache_reused"])
            self.assertEqual(first["choices"], second["choices"])

    def test_invalid_cached_score_is_removed_before_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text('{}')
            calls = []
            async def respond(*args):
                calls.append(1)
                if len(calls) == 1:
                    return {"choices": [{"message": {"content": '{}'}}], "_cache_path": str(path)}
                self.assertFalse(path.exists())
                return {"choices": [{"message": {"content": '{"score":1}'}}]}
            settings = SimpleNamespace(provider="doubao", max_retries=1)
            with patch("reward_as_agent.llm.call_llm", respond):
                result, api = asyncio.run(retry_llm_call([], 0, lambda obj: obj.get('score', -1), settings))
            self.assertEqual(api['score'], 1)
            self.assertEqual(len(calls), 2)

    def test_preserves_prompt_roles_images_and_reflection(self):
        messages = [{"role": "system", "content": "score"},
                    {"role": "user", "content": [{"type": "text", "text": "task"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}]},
                    {"role": "assistant", "content": '{"score":1}'},
                    {"role": "user", "content": "reflect"}]
        result = responses_input(messages)
        self.assertEqual([r["role"] for r in result], ["system", "user", "assistant", "user"])
        self.assertEqual(result[1]["content"], [{"type": "input_text", "text": "task"}, {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}])
        self.assertEqual(result[2:], messages[2:])

    def test_real_transport_shape_and_response_extraction(self):
        seen = []
        def respond(request):
            seen.append(request)
            return httpx.Response(200, json={"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": '{"score":1}'}]}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        settings = SimpleNamespace(api_base="https://example.invalid/api/v3", model="test", max_tokens=4096, max_retries=0, llm_timeout=10)
        with patch.dict(os.environ, {"ARK_API_KEY": "test-only"}), patch("reward_as_agent.doubao.httpx.AsyncClient", return_value=client):
            body = asyncio.run(call_doubao([{"role": "user", "content": "test"}], settings))
        self.assertEqual(body["choices"][0]["message"]["content"], '{"score":1}')
        self.assertEqual(str(seen[0].url), "https://example.invalid/api/v3/responses")
        self.assertNotIn("X-data-parallel-rank", seen[0].headers)

    def test_malformed_scoring_is_error_not_zero_reward(self):
        async def invalid(*args):
            return {"choices": [{"message": {"content": "not JSON"}}]}
        settings = SimpleNamespace(provider="doubao", max_retries=0)
        with patch("reward_as_agent.llm.call_llm", invalid):
            with self.assertRaises(RuntimeError):
                asyncio.run(retry_llm_call([], 0, lambda obj: -1, settings))


if __name__ == "__main__":
    unittest.main()
