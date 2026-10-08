"""Independent tool evidence followed by one image-grounded report reflection."""
import asyncio
import copy
import hashlib
import json

from scripts.frozen_v41.tool_protocol import video_sha256


class PhysicsEvidenceHook:
    def __init__(self, clients, mode='shadow', request_specs=None):
        if mode not in {'shadow', 'reflect'} or not clients:
            raise ValueError('Supply workers and an explicit shadow/reflect mode')
        self.clients = dict(clients)
        self.mode = mode
        self.request_specs = copy.deepcopy(request_specs or {})
        if set(self.request_specs) - set(self.clients):
            raise ValueError('Query specification names an unknown worker')
        for name in self.clients:
            queries = self.request_specs.setdefault(name, [{'operation': 'check_physics'}])
            if not isinstance(queries, list) or not queries:
                raise ValueError('Each worker requires at least one query')
            for query in queries:
                if not isinstance(query, dict) or set(query) - {'operation', 'rule', 'params', 'entities'}:
                    raise ValueError('Queries cannot override source identity or unsupported fields')
                if query.get('operation') not in {'check_physics', 'check_rule', 'inspect_interaction'}:
                    raise ValueError('Unsupported physical query')
                if query['operation'] == 'inspect_interaction':
                    from interaction_worker import entity_seeds
                    entity_seeds(query)
                elif 'entities' in query:
                    raise ValueError('Entity seeds require inspect_interaction')
                if query['operation'] == 'check_rule' and (
                        not isinstance(query.get('rule'), str) or not query['rule'].strip()):
                    raise ValueError('Rule queries require explicit rule text')

    async def apply(self, *, pipeline, video_path, video_sha256_expected, report,
                    contract, description, manifest, frames, valid_ids, validator, trace,
                    input_visibility=None):
        before = copy.deepcopy(report)
        request = {'video_uri': str(video_path), 'video_sha256': video_sha256_expected,
                   'operation': 'check_physics'}
        records = []
        # Keep execution errors in the record, including an individual worker crash.
        queries = [(name, client, spec) for name, client in self.clients.items()
                   for spec in self.request_specs[name]]
        for name, client, spec in queries:
            current_request = {**copy.deepcopy(request), **copy.deepcopy(spec)}
            try:
                result = await client.evaluate(copy.deepcopy(current_request))
            except Exception as exc:
                result = {'status': 'error', 'raw_score': None,
                          'error_type': type(exc).__name__, 'error': str(exc)}
            if (result.get('status') == 'ok' and
                    result.get('video_sha256') != video_sha256_expected):
                result = {'status': 'error', 'raw_score': None, 'rejected_output': result,
                          'error_type': 'SourceMismatch'}
            records.append({'worker': name, 'request': copy.deepcopy(current_request),
                            'result': copy.deepcopy(result)})
        if await asyncio.to_thread(video_sha256, video_path) != video_sha256_expected:
            raise ValueError('Source video changed during physics evaluation')
        raw = json.dumps(records, ensure_ascii=False, sort_keys=True, allow_nan=False)
        record = {'mode': self.mode, 'tool_records': records,
                  'input_visibility': copy.deepcopy(input_visibility),
                  'tool_records_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                  'pre_tool_report': before, 'fallback_to_baseline': False,
                  'reflection_applied': False}
        trace.append({'stage': 'independent_physics_tools_started',
                      'tool_records': copy.deepcopy(records),
                      'tool_records_sha256': record['tool_records_sha256']})
        def finish(updated):
            # Record the actual decision, including reflection failures and
            # abstention. An early snapshot falsely reported every reflection
            # as unapplied even when the final evidence said it succeeded.
            trace.append({'stage': 'independent_physics_tools', **copy.deepcopy(record)})
            return updated, record
        successful = [r for r in records if r['result'].get('status') == 'ok']
        if input_visibility and input_visibility['all_frames_spatially_uniform']:
            record.update(fallback_to_baseline=True, input_unobservable=True,
                          reflection_skip_reason='No spatial image evidence exists in decoded frames')
            return finish(before)
        if self.mode == 'shadow' or not successful:
            record['fallback_to_baseline'] = not successful
            return finish(before)
        from reward_as_agent.evidence_prompts import verification_prompt
        prompt = verification_prompt(manifest, task_contract=contract) + '''
Independent physical models are additional fallible evidence, not human labels.
Recheck the existing report against the supplied original video images. A tool's
status=ok means execution succeeded, not that physics is plausible. VideoPhy PC
uses 1..5 (higher is better); WMReward surprise is mean(1-cosine), lower is better,
not a calibrated probability or a severity category. VideoPhy Rule labels mean
0=violation, 1=adherence, 2=rule not grounded (abstain). Do not invent thresholds,
multiply scores, or change the frozen task requirements. Whole-video scores do
not locate defects. Every claimed event and severity needs visible source-frame
evidence; distinguish camera motion, occlusion and motion blur from anomalies.
Error/abstain records provide no positive or negative physical evidence. If the
tools do not resolve the issue, retain the existing judgment and uncertainty.
Return the complete grounded report in the existing schema. Do not rewrite tool
outputs or quote task intentions as observations.
Interaction masks and tracks are conditional image-plane measurements, not scores.
Seed roles and queries are requested identities, not visually established facts.
Tracks can drift while marked visible; small surviving-point motion does not
establish continuity when tracks disappear or leave the corresponding mask.
Camera motion remains unresolved unless an explicit measurement says otherwise.
'''
        try:
            updated = await pipeline.stage('physics_tool_reflection', prompt,
                {'task_description': description, 'task_contract': contract,
                 'existing_report': copy.deepcopy(before), 'independent_tool_records': copy.deepcopy(records)},
                frames, valid_ids, validator, trace)
            validator(updated, valid_ids)
            record['reflection_applied'] = True
            return finish(updated)
        except Exception as exc:
            record.update(fallback_to_baseline=True, reflection_error_type=type(exc).__name__,
                          reflection_error=str(exc))
            return finish(before)
