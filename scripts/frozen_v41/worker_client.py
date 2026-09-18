"""Serialized JSONL transport for expensive resident physics workers."""
import asyncio
import json
import math
from pathlib import Path


class WorkerClient:
    def __init__(self, command, stderr_path, timeout_s=600, startup_timeout_s=None):
        if not command or isinstance(command, str):
            raise ValueError('Worker command must be an argv sequence')
        self.command = list(command)
        self.stderr_path = Path(stderr_path)
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.process = None
        self._stderr = None
        self._lock = asyncio.Lock()

    async def _start(self):
        if self.process is not None and self.process.returncode is None:
            return
        await self._stop()
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open('ab')
        self.process = await asyncio.create_subprocess_exec(*self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr, limit=8 * 1024 * 1024)

    async def _stop(self):
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    async def _exchange(self, request):
        await self._start()
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode() + b'\n'
        self.process.stdin.write(payload)
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError('Worker exited before returning a result')
        def invalid_constant(value):
            raise ValueError('Nonfinite JSON constant: ' + value)
        result = json.loads(line, parse_constant=invalid_constant)
        if not isinstance(result, dict) or result.get('status') not in {'ok', 'error', 'abstain'}:
            raise ValueError('Invalid worker result status')
        score = result.get('raw_score')
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Invalid worker numeric score')
        if result['status'] != 'ok' and score is not None:
            raise ValueError('Failure/abstention must not carry a reward')
        if result['status'] == 'ok' and result.get('video_sha256') != request.get('video_sha256'):
            raise ValueError('Worker response source does not match request')
        return result

    async def evaluate(self, request):
        # The lock prevents crossing results between concurrent video evaluations.
        async with self._lock:
            cold = self.process is None or self.process.returncode is not None
            deadline = self.startup_timeout_s if cold and self.startup_timeout_s is not None else self.timeout_s
            try:
                result = await asyncio.wait_for(self._exchange(request), deadline)
                result['transport'] = {'cold_start_request':cold,'timeout_s':deadline}
                return result
            except asyncio.CancelledError:
                # A late response must never be consumed by the next request.
                await self._stop()
                raise
            except Exception as exc:
                await self._stop()
                return {'status': 'error', 'raw_score': None,
                        'error_type': type(exc).__name__, 'error': str(exc),
                        'failure_stage': 'worker_transport', 'fallback_to_baseline': True,
                        'transport': {'cold_start_request':cold,'timeout_s':deadline}}

    async def close(self):
        async with self._lock:
            await self._stop()
