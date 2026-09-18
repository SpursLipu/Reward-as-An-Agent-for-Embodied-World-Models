import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reward_as_agent.config import ConfigurationError
from reward_as_agent.tool_runtime import RequiredPhysicsHook, configured_hook
from scripts.frozen_v41.physics_integration import PhysicsEvidenceHook


def test_missing_tool_configuration_fails(monkeypatch, tmp_path):
    for key in ('PYTHON', 'REPO', 'CHECKPOINT', 'SHA256'):
        monkeypatch.delenv('REWARD_WMREWARD_' + key, raising=False)
    with pytest.raises(ConfigurationError, match='WMReward is required'):
        configured_hook(SimpleNamespace(log_root=tmp_path))


def test_configured_worker_is_required_and_uses_separate_interpreter(monkeypatch, tmp_path):
    repo = tmp_path / 'WMReward'
    (repo / 'vjepa2/src/hub').mkdir(parents=True)
    (repo / 'utils.py').touch()
    (repo / 'vjepa2/src/hub/backbones.py').touch()
    checkpoint = tmp_path / 'vitg.pt'
    checkpoint.touch()
    worker_python = tmp_path / 'venv/bin/python'
    worker_python.parent.mkdir(parents=True)
    worker_python.symlink_to(sys.executable)
    for name, value in {'PYTHON': str(worker_python), 'REPO': str(repo),
                        'CHECKPOINT': str(checkpoint), 'SHA256': 'a' * 64,
                        'DEVICE': 'cuda:1'}.items():
        monkeypatch.setenv('REWARD_WMREWARD_' + name, value)
    hook, client = configured_hook(SimpleNamespace(log_root=tmp_path))
    assert isinstance(hook, RequiredPhysicsHook)
    assert hook.mode == 'reflect'
    assert hook.clients == {'wmreward': client}
    assert client.command[-1] == 'cuda:1'
    assert client.command[0] == str(worker_python)
    assert client.process is None  # No GPU work merely from reading configuration.


@pytest.mark.parametrize('evidence', [
    {},
    {'tool_records': [{'result': {'status': 'error'}}]},
    {'tool_records': [{'result': {'status': 'abstain'}}]},
    {'tool_records': [{'result': {'status': 'ok'}}], 'reflection_applied': False},
    {'tool_records': [{'result': {'status': 'ok'}}], 'reflection_applied': True,
     'fallback_to_baseline': True},
])
def test_tool_failures_never_return_baseline(monkeypatch, evidence):
    monkeypatch.setattr(PhysicsEvidenceHook, 'apply', AsyncMock(return_value=({}, evidence)))
    hook = RequiredPhysicsHook({'wmreward': object()}, mode='reflect')
    with pytest.raises(RuntimeError, match='did not complete'):
        asyncio.run(hook.apply())


def test_successful_tool_reflection_preserves_report(monkeypatch):
    evidence = {'tool_records': [{'result': {'status': 'ok'}}], 'reflection_applied': True}
    report = {'unchanged': True}
    monkeypatch.setattr(PhysicsEvidenceHook, 'apply', AsyncMock(return_value=(report, evidence)))
    hook = RequiredPhysicsHook({'wmreward': object()}, mode='reflect')
    assert asyncio.run(hook.apply()) == (report, evidence)


def test_service_lifespan_loads_and_closes_worker(monkeypatch):
    from reward_as_agent import app as service
    from reward_as_agent import tool_runtime
    client = SimpleNamespace(evaluate=AsyncMock(return_value={'status': 'ok'}), close=AsyncMock())
    hook = object()
    monkeypatch.setattr(tool_runtime, 'configured_hook', lambda settings: (hook, client))
    monkeypatch.setattr(service, 'EvidencePipeline', lambda settings, physics_hook: physics_hook)

    async def check():
        async with service.lifespan(service.app):
            assert service.PIPELINE is hook
        assert service.PIPELINE is None
    asyncio.run(check())
    client.evaluate.assert_awaited_once_with({'operation': 'health'})
    client.close.assert_awaited_once()


def test_service_startup_fails_when_worker_fails(monkeypatch):
    from reward_as_agent import app as service
    from reward_as_agent import tool_runtime
    client = SimpleNamespace(evaluate=AsyncMock(return_value={'status': 'error'}), close=AsyncMock())
    monkeypatch.setattr(tool_runtime, 'configured_hook', lambda settings: (object(), client))

    async def check():
        async with service.lifespan(service.app):
            pytest.fail('Service must not start without its worker')
    with pytest.raises(RuntimeError, match='startup failed'):
        asyncio.run(check())
    client.close.assert_awaited_once()
