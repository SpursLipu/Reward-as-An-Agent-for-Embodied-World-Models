"""Run an isolated reward candidate from a frozen video manifest and worker config."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
# The frozen archive kept the candidate package below the runner.  In this
# repository the package is installed at the repository root.
sys.path.insert(0, str(ROOT.parent))

from scripts.frozen_v41.physics_integration import PhysicsEvidenceHook
from scripts.frozen_v41.tool_protocol import video_sha256
from scripts.frozen_v41.worker_client import WorkerClient
from scripts.frozen_v41.replay_client import ReplayClient


async def run(args):
    from reward_as_agent.config import get_settings
    from reward_as_agent.evidence_pipeline import EvidencePipeline
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = Path(args.manifest).read_bytes()
    rows = json.loads(manifest_bytes)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty list')
    required = {'sample_id', 'video_path', 'description', 'video_sha256'}
    if any(not isinstance(r, dict) or not required <= r.keys() for r in rows):
        raise ValueError('Manifest requires sample_id/video_path/description/video_sha256')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    if not os.environ.get('REWARD_TASK_CONTRACT_REGISTRY'):
        raise ValueError('A frozen task-contract registry is required for comparison')
    clients = {}
    settings = get_settings()
    config = None
    if args.mode != 'baseline':
        if not args.workers:
            raise ValueError('Tool modes require --workers JSON configuration')
        config = json.loads(Path(args.workers).read_text())
        for name, spec in config.items():
            if not name or Path(name).name != name:
                raise ValueError('Invalid worker name')
            if 'replay_results' in spec:
                if 'command' in spec:
                    raise ValueError('Choose live or replay worker, not both')
                clients[name] = ReplayClient(spec['replay_results'], spec.get('repeat', 0))
            else:
                clients[name] = WorkerClient(spec['command'], output/(name+'.stderr.log'),
                                             timeout_s=spec.get('timeout_s', 900),
                                             startup_timeout_s=spec.get('startup_timeout_s'))
    hook = PhysicsEvidenceHook(clients, args.mode,
        {name: spec.get('requests', [{'operation': 'check_physics'}])
         for name, spec in config.items()}) if clients else None
    event_client=None;event_hook=None
    if getattr(args,'event_worker',None):
        from process_event_integration import GroundedProcessEventHook
        event_config=json.loads(Path(args.event_worker).read_text())
        event_client=WorkerClient(event_config['command'],output/'event_worker.stderr.log',
            timeout_s=event_config.get('timeout_s',600),startup_timeout_s=event_config.get('startup_timeout_s',900))
        event_hook=GroundedProcessEventHook(event_client,output/'event_evidence')
    pipeline = EvidencePipeline(settings, physics_hook=hook,process_event_hook=event_hook)
    frozen = {'mode': args.mode, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'implementation_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(ROOT.glob('*.py'))},
              'evaluator_version': pipeline.evaluator_version,
              'event_hook_fingerprint': event_hook.fingerprint if event_hook else None,
              'worker_config_sha256': (hashlib.sha256(Path(args.workers).read_bytes()).hexdigest()
                                       if config is not None else None),
              'repeats': args.repeats, 'human_labels': 0}
    (output/'run_config.json').write_text(json.dumps(frozen, indent=2))
    (output/'input_manifest.json').write_bytes(manifest_bytes)
    try:
        with (output/'results.jsonl').open('x') as stream:
            for repeat in range(args.repeats):
                for index, row in enumerate(rows):
                    started = time.monotonic()
                    result = {'sample_id': row['sample_id'], 'repeat': repeat,
                              'status': 'error', 'new_score': None}
                    try:
                        os.environ['REWARD_EVIDENCE_TRACE_DIR'] = str(output/'traces'/str(repeat))
                        if await asyncio.to_thread(video_sha256, row['video_path']) != row['video_sha256']:
                            raise ValueError('Frozen input SHA256 mismatch')
                        details = await pipeline.process_one_video(row['video_path'], row['description'], index)
                        if await asyncio.to_thread(video_sha256, row['video_path']) != row['video_sha256']:
                            raise ValueError('Source video changed during evaluation')
                        result.update(status='success' if details['training_eligible'] else 'needs_review',
                                      new_score=details['scoring']['total_score'], details=details)
                    except Exception as exc:
                        result.update(error_type=type(exc).__name__, error=str(exc))
                    result['elapsed_seconds'] = time.monotonic() - started
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({k: result[k] for k in ('sample_id', 'repeat', 'status', 'new_score')}), flush=True)
    finally:
        for client in clients.values():
            await client.close()
        if event_client is not None:await event_client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['baseline', 'shadow', 'reflect'], required=True)
    parser.add_argument('--workers')
    parser.add_argument('--event-worker',help='Optional actual local-event worker JSON; requires evidence_process policy')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
