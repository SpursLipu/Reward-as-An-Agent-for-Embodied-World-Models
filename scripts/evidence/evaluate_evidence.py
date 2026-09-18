"""Resumable evidence-v2 evaluation; abstentions are separate from numerical rewards."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from reward_as_agent.config import get_settings
from reward_as_agent.evidence_pipeline import EvidencePipeline, PIPELINE_VERSION


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


async def run(args):
    rows = json.loads(args.manifest.read_text()) if args.manifest.suffix == '.json' else [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if isinstance(rows, dict):
        rows = rows.get('samples', rows.get('rows'))
    if not isinstance(rows, list):
        raise ValueError('Manifest must contain a list of sample records')
    if args.split:
        rows = [r for r in rows if r.get('split') == args.split]
    if args.limit:
        rows = rows[:args.limit]
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest contains duplicate sample IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / 'results.jsonl'
    prior = {r['sample_id']: r for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else {}
    completed = {sid for sid,r in prior.items() if r['status'] in ('success', 'needs_review')}
    pipeline = EvidencePipeline(get_settings())
    config_path=args.output/'run_config.json'
    config={'evaluator_version':pipeline.evaluator_version,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'task_contract_registry_sha256':pipeline.contract_registry_sha256,
            'requested_sampling': {'temperature': pipeline.settings.temperature,
                                   'temperature_in_payload': pipeline.settings.temperature is not None,
                                   'top_p_in_payload': False, 'thinking': 'disabled'},
            'split':args.split,'limit':args.limit,'cache_bypass':os.environ.get('REWARD_AS_AGENT_CACHE_BYPASS')=='1'}
    if config_path.exists():
        existing=json.loads(config_path.read_text())
        if any(existing.get(k)!=v for k,v in config.items()):
            raise ValueError('Run configuration changed; use a new output directory to preserve comparability')
        run_id=existing['run_id']
    else:
        run_id=str(uuid.uuid4())
        config_path.write_text(json.dumps({**config,'run_id':run_id},indent=2))
    pending = [r for r in rows if r['sample_id'] not in completed]
    slots = asyncio.Semaphore(args.jobs)
    count = 0
    errors = 0
    started = time.monotonic()
    async def evaluate(row):
        nonlocal count, errors
        async with slots:
            t = time.monotonic()
            base={k:row[k] for k in ('sample_id','video_path','prompt','old_score','stratum','weight','split','audit_id') if k in row}
            base.update(baseline_doubao_score=row.get('new_score'),run_id=run_id,evaluator_version=pipeline.evaluator_version)
            try:
                base['source_video_sha256'] = await asyncio.to_thread(file_sha256, row['video_path'])
                details = await pipeline.process_one_video(row['video_path'], row['prompt'], 0)
                if await asyncio.to_thread(file_sha256, row['video_path']) != base['source_video_sha256']:
                    raise ValueError('Source video changed during evaluation; discard this result')
                status = 'needs_review' if details['review_required'] else 'success'
                record = {**base, 'status': status, 'new_score': details['scoring']['total_score'],
                          'task_contract_sha256': details['task_contract_sha256'],
                          'training_eligible': details['training_eligible'], 'details': details,
                          'cache_reused':any(t.get('cache_reused',False) for t in details['trace']),
                          'pipeline_version': PIPELINE_VERSION}
            except Exception as exc:
                errors += 1
                record = {**base, 'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc),
                          'pipeline_version': PIPELINE_VERSION}
            record['elapsed_seconds'] = round(time.monotonic()-t, 3)
            with dest.open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
            print(json.dumps({'completed':count, 'total':len(pending), 'sample_id':row['sample_id'],
                              'new_score':record.get('new_score'), 'status':record['status'],
                              'error':record.get('error'), 'seconds':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
    await asyncio.gather(*(evaluate(r) for r in pending))
    return 1 if errors else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--split')
    args=parser.parse_args()
    if args.jobs<1 or args.jobs>4:
        parser.error('jobs must be between 1 and 4')
    raise SystemExit(asyncio.run(run(args)))
