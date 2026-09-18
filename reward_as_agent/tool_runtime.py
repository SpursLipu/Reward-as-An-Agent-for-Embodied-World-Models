"""Required WMReward runtime shared by the public service and demo runners."""
import os
from pathlib import Path
import re

from reward_as_agent.config import ConfigurationError

from scripts.frozen_v41.physics_integration import PhysicsEvidenceHook
from scripts.frozen_v41.worker_client import WorkerClient


class RequiredPhysicsHook(PhysicsEvidenceHook):
    async def apply(self, **kwargs):
        report, evidence = await super().apply(**kwargs)
        records = evidence.get('tool_records', [])
        if (not records or any(r['result'].get('status') != 'ok' for r in records)
                or not evidence.get('reflection_applied')
                or evidence.get('fallback_to_baseline')):
            raise RuntimeError('Required WMReward/Reflection did not complete; see tool trace and worker log')
        return report, evidence


def configured_hook(settings):
    names = ('REWARD_WMREWARD_PYTHON', 'REWARD_WMREWARD_REPO',
             'REWARD_WMREWARD_CHECKPOINT', 'REWARD_WMREWARD_SHA256')
    values = [os.environ.get(name, '').strip() for name in names]
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ConfigurationError('WMReward is required. Configure: ' + ', '.join(missing))
    python, repo, checkpoint = [Path(value).expanduser().resolve() for value in values[:3]]
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ConfigurationError('REWARD_WMREWARD_PYTHON must be an executable interpreter')
    if not (repo / 'utils.py').is_file() or not (repo / 'vjepa2/src/hub/backbones.py').is_file():
        raise ConfigurationError('REWARD_WMREWARD_REPO requires WMReward and its vjepa2 submodule')
    if not checkpoint.is_file() or not re.fullmatch('[0-9a-f]{64}', values[3]):
        raise ConfigurationError('WMReward checkpoint file and SHA256 digest are required')
    # -c adds the installed adapter location even with a separate CUDA interpreter.
    package_root = str(Path(__file__).resolve().parents[1])
    bootstrap = ('import runpy,sys; sys.path.insert(0, ' + repr(package_root) + '); '
                 'runpy.run_module("scripts.frozen_v41.wmreward_worker", run_name="__main__")')
    command = [str(python), '-c', bootstrap, '--repo', str(repo),
               '--checkpoint', str(checkpoint), '--checkpoint-sha256', values[3],
               '--device', os.environ.get('REWARD_WMREWARD_DEVICE', 'cuda:0')]
    client = WorkerClient(command, settings.log_root / 'wmreward.stderr.log',
                          timeout_s=900, startup_timeout_s=900)
    return RequiredPhysicsHook({'wmreward': client}, mode='reflect'), client
