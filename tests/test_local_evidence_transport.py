import asyncio
import hashlib
import json
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from reward_as_agent.config import get_settings
from reward_as_agent.evidence_pipeline import EvidencePipeline
from reward_as_agent.llm import call_llm


def test_openai_pipeline_preserves_prompt_budget_and_distinguishes_provider(monkeypatch):
    monkeypatch.setenv('REWARD_TASK_CONTRACT_REGISTRY', '')
    settings = replace(get_settings(), provider='openai', max_tokens=1024)
    local = EvidencePipeline(settings)
    ark = EvidencePipeline(replace(settings, provider='doubao'))
    assert local.settings.max_tokens == 8192
    assert local.evaluator_version != ark.evaluator_version


def test_local_transport_records_exact_payload_and_retains_images():
    settings = replace(get_settings(), provider='openai', dp_size=1)
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Return JSON'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}
    ]}]
    client = AsyncMock()
    from unittest.mock import Mock
    client.post.return_value = Mock(json=lambda: {'choices': [{'message': {'content': '{}'}}]})
    with patch('reward_as_agent.llm.httpx.AsyncClient') as factory:
        factory.return_value.__aenter__.return_value = client
        result = asyncio.run(call_llm(messages, settings))
    body = client.post.call_args.kwargs['content']
    payload = json.loads(body)
    assert payload['messages'] == messages
    assert payload['chat_template_kwargs']['enable_thinking'] is False
    assert result['request_payload_sha256'] == hashlib.sha256(body).hexdigest()
    assert result['request_sampling']['temperature']['value'] == 0
    factory.assert_called_once_with(trust_env=False)
