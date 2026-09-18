import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from reward_as_agent.doubao import call_doubao, responses_input
from reward_as_agent.llm import retry_llm_call


class DoubaoTests(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"ARK_API_KEY": "test-only", "REWARD_AS_AGENT_CACHE_DIR": folder}), patch("reward_as_agent.doubao.httpx.AsyncClient", return_value=client):
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
