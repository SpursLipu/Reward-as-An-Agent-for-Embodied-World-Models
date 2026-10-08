"""Required WMReward runtime shared by the public service and demo runners."""
import os
from pathlib import Path
import re

from reward_as_agent.config import ConfigurationError

from scripts.frozen_v41.physics_integration import PhysicsEvidenceHook
from scripts.frozen_v41.worker_client import WorkerClient
import asyncio


class WorkerPool:
    """Bounded round-robin pool of resident WMReward workers."""
    def __init__(self, clients):
        self.clients = list(clients)
        if not self.clients:
            raise ValueError('WMReward worker pool cannot be empty')
        self._slots = asyncio.Semaphore(len(self.clients))
        self._lock = asyncio.Lock()
        self._next = 0

    async def evaluate(self, request):
        await self._slots.acquire()
        try:
            async with self._lock:
                client = self.clients[self._next % len(self.clients)]
                self._next += 1
            return await client.evaluate(request)
        finally:
            self._slots.release()

    async def close(self):
        await asyncio.gather(*(client.close() for client in self.clients), return_exceptions=True)

    async def warmup(self):
        """Start every resident worker before the API reports ready."""
        results = await asyncio.gather(
            *(client.evaluate({'operation': 'health'}) for client in self.clients),
            return_exceptions=True,
        )
        failures = [result for result in results
                    if isinstance(result, BaseException)
                    or result.get('status') != 'ok']
        if failures:
            raise RuntimeError(f'WMReward worker warmup failed: {failures!r}')


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
    # Preserve venv Python symlinks: resolving them would select the base
    # interpreter and lose the CUDA environment's installed dependencies.
    python = Path(values[0]).expanduser().absolute()
    repo, checkpoint = [Path(value).expanduser().resolve() for value in values[1:3]]
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
    count = int(os.environ.get('REWARD_WMREWARD_WORKERS', '1'))
    if count != 1:
        raise ConfigurationError(
            'REWARD_WMREWARD_WORKERS must be 1 unless distinct CUDA devices are configured; '
            'multiple workers on one visible device can change resource behavior')
    clients = []
    for index in range(count):
        command = [str(python), '-c', bootstrap, '--repo', str(repo),
                   '--checkpoint', str(checkpoint), '--checkpoint-sha256', values[3],
                   '--device', os.environ.get('REWARD_WMREWARD_DEVICE', 'cuda:0')]
        clients.append(WorkerClient(
            command, settings.log_root / f'wmreward_{index}.stderr.log',
            timeout_s=900, startup_timeout_s=900))
    # Keep the single-worker API identical to the original runtime. This
    # avoids an extra scheduling layer and preserves callers that inspect the
    # WorkerClient command before startup; a pool is only meaningful when
    # distinct devices are explicitly supported.
    client = clients[0]
    return RequiredPhysicsHook({'wmreward': client}, mode='reflect'), client
