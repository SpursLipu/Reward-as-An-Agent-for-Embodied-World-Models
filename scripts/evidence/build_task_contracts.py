"""Prepare and audit task text before any video judgement is run."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from reward_as_agent.task_contract import validate_task_contract
from scripts.evidence.check_diagnostics import read_rows


def load_runtime():
    # Seed validation and offline tests need no model/HTTP dependencies.
    from reward_as_agent.config import get_settings
    from reward_as_agent.evidence_pipeline import EvidencePipeline, PersistentTrace
    return EvidencePipeline, get_settings, PersistentTrace


def load_seed_registry(path, sources):
    """Validate the whole seed, then retain only exact current-source matches."""
    raw = path.read_bytes()
    seed = json.loads(raw)
    if seed.get('schema_version') != 'task-contract-registry-v1':
        raise ValueError('Unsupported seed registry schema')
    all_contracts = seed.get('contracts')
    if not isinstance(all_contracts, dict):
        raise ValueError('Seed contracts must be an object')
    known = {}
    imported = {}
    for key, contract in all_contracts.items():
        validate_task_contract(contract)
        if key != contract['source_sha256']:
            raise ValueError('Seed registry key does not match source hash')
        known[contract['contract_sha256']] = contract
        if key in sources:
            validate_task_contract(contract, sources[key])
            imported[key] = contract
    reviews = seed.get('contract_reviews', {})
    if not isinstance(reviews, dict):
        raise ValueError('Seed contract_reviews must be an object')
    imported_reviews = {}
    for key, review in reviews.items():
        if not isinstance(review, dict) or key not in known or review.get('source_sha256') != known[key]['source_sha256']:
            raise ValueError('Seed review does not identify its frozen contract')
        if review.get('status') != 'requires_review' or not isinstance(review.get('reason'), str) or not review['reason'].strip():
            raise ValueError('Seed review must retain its unresolved interpretation reason')
        if known[key]['source_sha256'] in imported:
            if review.get('human_label') is not False or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
                raise ValueError('Imported seed review must identify its reviewer and explicitly set human_label=false')
            imported_reviews[key] = review
    provenance = {
        'seed_registry_sha256': hashlib.sha256(raw).hexdigest(),
        'seed_registry_path': str(path.resolve()),
        'seed_preparation_evaluator_version': seed.get('preparation_evaluator_version'),
        'imported_count': len(imported),
        'imported_contracts': {key: value['contract_sha256'] for key, value in sorted(imported.items())},
        'imported_review_count': len(imported_reviews),
    }
    return imported, imported_reviews, provenance


def check_seed_conflicts(output, imported):
    """Preflight every seed destination before overwriting any existing artifact."""
    registry_path = output / 'registry.json'
    existing = json.loads(registry_path.read_text()).get('contracts', {}) if registry_path.exists() else {}
    for key, contract in imported.items():
        dest = output / 'contracts' / (key + '.json')
        candidates = [json.loads(dest.read_text())] if dest.exists() else []
        if key in existing:
            candidates.append(existing[key])
        for candidate in candidates:
            validate_task_contract(candidate, contract['source_text'])
            if candidate != contract:
                raise ValueError(f'Seed conflicts with existing frozen contract for source {key}; refusing overwrite')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temp.replace(path)


async def run(args):
    if os.environ.get('REWARD_TASK_CONTRACT_REGISTRY'):
        raise ValueError('Contract preparation must run without an existing registry selected')
    rows = read_rows(args.manifest)
    sources = {hashlib.sha256(row['prompt'].encode()).hexdigest(): row['prompt'] for row in rows}
    imported, imported_reviews, seed_provenance = {}, {}, None
    seed_path = getattr(args, 'seed_registry', None)
    if seed_path is not None:
        imported, imported_reviews, seed_provenance = load_seed_registry(seed_path, sources)
        check_seed_conflicts(args.output, imported)
    EvidencePipeline, get_settings, PersistentTrace = load_runtime()
    pipeline = EvidencePipeline(get_settings())
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {'preparation_evaluator_version': pipeline.evaluator_version,
                     'source_manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                     'cache_bypass': os.environ.get('REWARD_AS_AGENT_CACHE_BYPASS') == '1'}
    if seed_provenance is not None:
        configuration['seed_registry_sha256'] = seed_provenance['seed_registry_sha256']
    config_path = args.output / 'preparation_config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != configuration:
            raise ValueError('Existing preparation requires --resume and identical configuration')
    else:
        write_json(config_path, configuration)
    contracts, failures = dict(imported), []
    for key, contract in imported.items():
        dest = args.output / 'contracts' / (key + '.json')
        if not dest.exists():
            write_json(dest, contract)
    if seed_provenance is not None:
        print(json.dumps({'status': 'seed_imported', **seed_provenance,
                          'remaining_unique_sources': len(sources) - len(imported)}), flush=True)
    slots = asyncio.Semaphore(args.jobs)

    async def prepare(key, source):
        async with slots:
            dest = args.output / 'contracts' / (key + '.json')
            if dest.exists():
                contract = json.loads(dest.read_text())
                validate_task_contract(contract, source)
                contracts[key] = contract
                return
            trace = PersistentTrace(args.output / 'traces' / (key + '.json'),
                                    {'source_text': source, **configuration})
            try:
                contract = await pipeline.prepare_task_contract(source, trace)
                write_json(dest, contract)
                contracts[key] = contract
                print(json.dumps({'source_sha256': key, 'status': 'prepared',
                                  'requirements': len(contract['requirements'])}), flush=True)
            except Exception as exc:
                failures.append(key)
                print(json.dumps({'source_sha256': key, 'status': 'error',
                                  'error': type(exc).__name__ + ': ' + str(exc)}), flush=True)

    await asyncio.gather(*(prepare(key, source) for key, source in sources.items() if key not in imported))
    if failures:
        return 1
    registry = {'schema_version': 'task-contract-registry-v1', **configuration,
                'contracts': contracts}
    if seed_provenance is not None:
        registry['seed_import'] = seed_provenance
        registry['contract_reviews'] = imported_reviews
    write_json(args.output / 'registry.json', registry)
    ready = {'status': 'registry_ready', 'unique_tasks': len(contracts), 'manifest_samples': len(rows)}
    if seed_provenance is not None:
        ready['imported_tasks'] = len(imported)
    print(json.dumps(ready), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed-registry', type=Path,
                        help='Import exact matching frozen contracts and AI review flags; extract only remaining task texts')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error('jobs must be between 1 and 4')
    raise SystemExit(asyncio.run(run(args)))
