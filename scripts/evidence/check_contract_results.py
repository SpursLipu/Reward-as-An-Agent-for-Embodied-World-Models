#!/usr/bin/env python3
"""Offline contract/provenance/reducer checks, not visual or human accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reward_as_agent.task_contract import validate_task_contract, validate_requirement_checks
from reward_as_agent.evidence_grounding import (
    core_report, validate_grounded_report, observation_time_manifest,
)
from reward_as_agent.evidence_schema import score_report
from reward_as_agent.requirement_audit import validate_requirement_audit
from reward_as_agent.training_reward import apply_training_reward, validate_failure_resolution


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    text = raw.decode('utf-8').strip()
    if not text:
        return []
    rows = json.loads(text) if text.startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{path}: expected JSON array or JSONL objects')
    return rows


def same(actual, expected, name):
    # JSON equality also distinguishes true from 1, unlike Python equality.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f'{name}: does not equal recomputed/frozen value')


def check_record(row, source, contract, registry_hash, contract_review=None):
    """Raise on any invalid accepted output; never repair or replace a score."""
    same(row['prompt'], source['prompt'], 'prompt')
    same(row['video_path'], source['video_path'], 'video_path')
    details = row['details']
    validate_task_contract(details['task_contract'], source['prompt'])
    same(details['task_contract'], contract, 'details.task_contract')
    for container, label in ((row, 'result'), (details, 'details')):
        same(container['task_contract_sha256'], contract['contract_sha256'], label + '.task_contract_sha256')
    same(details['task_contract_registry_sha256'], registry_hash, 'task_contract_registry_sha256')
    same(details['task_contract_review'], contract_review, 'task_contract_review')
    report, frames = details['evidence_report'], details['frame_manifest']
    if not isinstance(frames, list) or not frames:
        raise ValueError('frame_manifest: missing actual supplied frames')
    metadata = details['video_metadata']
    fps, count = metadata['fps'], metadata['decoded_frames']
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0 or type(count) is not int or count <= 0:
        raise ValueError('video_metadata: invalid fps/decoded_frames')
    ids = []
    for frame in frames:
        frame_id = frame['source_frame_index']
        if type(frame_id) is not int or not 0 <= frame_id < count:
            raise ValueError('frame_manifest: invalid original frame ID')
        if frame_id in ids:
            raise ValueError('frame_manifest: duplicate original frame ID')
        ids.append(frame_id)
        same(frame['timestamp_seconds'], round(frame_id / fps, 4), 'frame timestamp from FPS')
    validate_grounded_report(report, ids, contract)
    summary = validate_requirement_checks(report['requirement_checks'], contract, report)
    same(details['requirement_summary'], summary, 'requirement_summary')
    same(details['observation_time_manifest'], observation_time_manifest(report, frames), 'observation_time_manifest')
    expected = score_report(core_report(report))
    expected['scoring_version'] = 'evidence-soft-physics-v2-with-scope-review-v5-provisional'
    for requirement_id in summary['unresolved_requirement_ids']:
        expected['review_reasons'].append(f'task requirement {requirement_id}: unresolved visible evidence')
    if contract_review:
        expected['review_reasons'].append('task contract: ' + contract_review['reason'])
    audits = details['requirement_scope_audits']
    if not isinstance(audits, list) or len(audits) not in (1, 2):
        raise ValueError('scope audit must run once, with at most one repair and re-audit')
    initial = details['pre_scope_repair_report']
    validate_grounded_report(initial, ids, contract)
    for index, entry in enumerate(audits):
        audited_report = initial if index == 0 else report
        same(entry['report_sha256'], digest(json.dumps(
            audited_report, ensure_ascii=False, sort_keys=True, separators=(',', ':')
        ).encode()), 'audited report hash')
        validate_requirement_audit(entry['audit'], contract, audited_report)
    initial_issues = any(check['issues'] for check in audits[0]['audit']['checks'])
    if (len(audits) == 2) != initial_issues:
        raise ValueError('scope audit issues must trigger exactly one repair and re-audit')
    if len(audits) == 1:
        same(report, initial, 'unaudited report must not replace the accepted report')
    for check in audits[-1]['audit']['checks']:
        for issue in check['issues']:
            expected['review_reasons'].append(
                f"task requirement {check['requirement_id']}: unresolved scope audit "
                f"({issue['kind']}): {issue['reason']}"
            )
    expected['review_required'] = bool(expected['review_reasons'])
    resolution = details.get('failure_reward_resolution')
    if resolution is not None:
        validate_failure_resolution(resolution, report, contract, audits, set(ids))
        accepted = [entry for entry in details.get('trace', [])
                    if entry.get('stage') == 'failure_reward_resolution'
                    and not entry.get('validation_error') and not entry.get('error')]
        if not accepted:
            raise ValueError('failure resolution: missing successful image-grounded trace')
        same(accepted[-1]['output'], resolution, 'failure resolution trace')
    expected = apply_training_reward(
        expected, report, contract, audits, contract_review=contract_review,
        physics_evidence=details.get('physics_evidence'), failure_resolution=resolution,
    )
    same(details['scoring'], expected, 'scoring')
    for field in ('diagnostic_score', 'diagnostic_review_required', 'diagnostic_review_reasons'):
        same(details[field], expected[field], field)
    score, review = expected['total_score'], expected['review_required']
    eligible = score is not None and not review
    same(row['new_score'], score, 'new_score')
    same(details['total_score'], score if score is not None else -1, 'transport total_score')
    same(details['review_required'], review, 'review_required')
    same(details['training_eligible'], eligible, 'details.training_eligible')
    same(row['training_eligible'], eligible, 'training_eligible')
    same(row['status'], 'needs_review' if review else 'success', 'status')
    return {'expected_score': score, 'expected_review_required': review,
            'expected_training_eligible': eligible, 'requirement_summary': summary}


def analyze(manifest_path, registry_path, results_paths):
    manifest = read_rows(manifest_path)
    sources = {row['sample_id']: row for row in manifest}
    if not sources or len(sources) != len(manifest):
        raise ValueError('manifest: empty or duplicate sample_id')
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if registry['schema_version'] != 'task-contract-registry-v1':
        raise ValueError('unsupported registry schema')
    contracts = registry['contracts']
    for key, contract in contracts.items():
        validate_task_contract(contract)
        same(key, contract['source_sha256'], 'registry key')
    for row in manifest:
        key = digest(row['prompt'].encode('utf-8'))
        validate_task_contract(contracts[key], row['prompt'])
    registry_hash = digest(registry_bytes)
    reports = []
    for path in results_paths:
        rows = read_rows(path)
        groups = {}
        for index, row in enumerate(rows, 1):
            groups.setdefault(row.get('run_id') or '__missing_run_id__', []).append((index, row))
        if not groups:
            groups['__empty_results__'] = []
        for run_id, records in groups.items():
            attempts, latest = [], {}
            for index, row in records:
                sid = row.get('sample_id')
                item = {'line': index, 'sample_id': sid, 'status': row.get('status'),
                        'check': 'unresolved', 'error': row.get('error')}
                if sid not in sources:
                    item.update(check='fail', error='sample_id not in manifest')
                elif row.get('status') in ('success', 'needs_review'):
                    try:
                        key = digest(sources[sid]['prompt'].encode('utf-8'))
                        contract_review = registry.get('contract_reviews', {}).get(contracts[key]['contract_sha256'])
                        item.update(check_record(row, sources[sid], contracts[key], registry_hash, contract_review))
                        item.update(check='pass', error=None)
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        item.update(check='fail', error=f'{type(exc).__name__}: {exc}')
                elif row.get('status') != 'error':
                    item.update(error='missing or unsupported status')
                attempts.append(item)
                if sid in sources:
                    latest[sid] = item
            for sid in sources:
                latest.setdefault(sid, {'sample_id': sid, 'status': 'missing', 'check': 'unresolved', 'error': 'no result'})
            counts = Counter(item['check'] for item in latest.values())
            reports.append({'results_file': str(path), 'run_id': run_id,
                            'manifest_count': len(sources),
                            'latest_counts': {key: counts[key] for key in ('pass', 'fail', 'unresolved')},
                            'attempt_check_counts': dict(Counter(item['check'] for item in attempts)),
                            'latest': list(latest.values()), 'all_attempts': attempts})
    all_valid = bool(reports) and all(
        run['latest_counts']['pass'] == len(sources)
        and all(item['check'] == 'pass' for item in run['all_attempts']) for run in reports)
    return {'schema_version': 'contract-result-diagnostics-v1',
            'registry_sha256': registry_hash, 'contract_count': len(contracts),
            'all_recorded_contract_checks_passed': all_valid,
            'human_accuracy_established': False, 'human_accuracy': None,
            'limits': ['Checks logged frame provenance and deterministic reducer consistency, not the images or natural-language truth.',
                       'Registry structural validity does not establish semantic coverage.',
                       'Repeated input files are separate audit inputs, not proof of independent video judgments.'],
            'runs': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        output = analyze(args.manifest, args.registry, args.results)
        code = 0 if output['all_recorded_contract_checks_passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        output = {'all_recorded_contract_checks_passed': False, 'input_error': f'{type(exc).__name__}: {exc}',
                  'human_accuracy_established': False, 'human_accuracy': None}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'all_recorded_contract_checks_passed': output['all_recorded_contract_checks_passed'], 'exit_code': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
