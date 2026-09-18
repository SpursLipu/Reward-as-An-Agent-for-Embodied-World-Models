"""Task-blind observation, shared assessment, and evidence-based verification."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
import re
from pathlib import Path
import time

from reward_as_agent.evidence_prompts import observation_prompt, assessment_prompt, verification_prompt
from reward_as_agent.evidence_schema import score_report
from reward_as_agent.evidence_grounding import (
    core_report, validate_grounded_report, validate_observation_time_format, observation_time_manifest,
)
from reward_as_agent.task_contract import (
    extraction_prompt, coverage_audit_prompt, validate_extraction, freeze_contract,
    validate_task_contract, validate_requirement_checks,
)
from reward_as_agent.evidence_video import load_evidence_video, uniform_indices, verification_indices
from reward_as_agent.llm import call_llm, safe_parse_json
from reward_as_agent.requirement_audit import audit_prompt, validate_requirement_audit
from reward_as_agent.focused_audit import focused_prompt, validate_focused_audit
from reward_as_agent.training_reward import (
    apply_training_reward, failure_candidates, failure_resolution_prompt,
    needs_failure_resolution, validate_failure_resolution,
)

PIPELINE_VERSION = 'evidence-v7.0-focused1'
# Preserve the v5 model-visible context; the candidate version is runtime provenance.
MODEL_CONTEXT_VERSION = 'evidence-v5.0-dev1'


def sampling_trace_fields(value):
    """Missing metadata stays unknown, notably for application caches from v5."""
    value = value if isinstance(value, dict) else {}
    return {field: value.get(field) for field in (
        'request_payload_sha256', 'request_sampling', 'response_sampling', 'sampling_metadata_status')}


class PersistentTrace(list):
    """Save completed stage evidence even when later requests fail."""
    def __init__(self, path=None, context=None):
        super().__init__()
        self.path = path
        self.context = context or {}

    def append(self, item):
        super().append(item)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.' + str(time.time_ns()) + '.tmp')
            temp.write_text(json.dumps({**self.context, 'status': 'error' if item.get('error') else 'running', 'trace': list(self)}, ensure_ascii=False, indent=2))
            temp.replace(self.path)


def validate_observations(value, valid_ids):
    if not isinstance(value, dict) or not isinstance(value.get('observations'), list):
        raise ValueError('observations must be a list')
    if set(value) != {'observations', 'uncertainties'}:
        raise ValueError('Blind output accepts only observations and uncertainties, not task judgements or scores')
    if not value['observations']:
        raise ValueError('At least one observation is required, including if only visibility can be described')
    seen = set()
    for obs in value['observations']:
        if not isinstance(obs, dict) or set(obs) != {'id', 'frames', 'description'}:
            raise ValueError('Observation fields must be exactly id, frames, description')
        if not isinstance(obs.get('id'), str) or not re.fullmatch(r'E[1-9][0-9]*', obs['id']) or obs['id'] in seen:
            raise ValueError('Observation IDs must be unique nonempty strings')
        seen.add(obs['id'])
        if not isinstance(obs.get('description'), str) or not obs['description'].strip():
            raise ValueError('Each observation needs a visible description')
        frames = obs.get('frames')
        if not isinstance(frames, list) or not frames or any(type(i) is not int or i not in valid_ids for i in frames):
            raise ValueError(f'Observation {obs["id"]} must cite actual supplied source frame IDs')
        if len(set(frames)) != len(frames):
            raise ValueError('Observation frame references must be unique')
    if not isinstance(value.get('uncertainties'), list) or any(not isinstance(x, str) or not x.strip() for x in value['uncertainties']):
        raise ValueError('uncertainties must be a list of strings')
    validate_observation_time_format(value)


class EvidencePipeline:
    def __init__(self, settings, physics_hook=None, process_event_hook=None):
        self.physics_hook = physics_hook
        self.process_event_hook = process_event_hook
        physical_policy = os.environ.get('REWARD_PHYSICS_COMPLETION', 'optional')
        if physical_policy not in {'optional', 'required'}:
            raise ValueError('REWARD_PHYSICS_COMPLETION must be optional or required')
        self.physics_completion_required = physical_policy == 'required'
        from reward_as_agent.process_gates import POLICIES
        self.gate_policy = os.environ.get('REWARD_GATE_POLICY', 'off')
        if self.gate_policy not in POLICIES:
            raise ValueError('Unknown REWARD_GATE_POLICY')
        if process_event_hook is not None and self.gate_policy!='evidence_process':
            raise ValueError('Grounded process events require REWARD_GATE_POLICY=evidence_process')
        if settings.provider != 'doubao':
            raise ValueError('evidence_v2 currently requires the validated Doubao transport')
        self.settings = replace(settings, max_tokens=max(settings.max_tokens, 8192))
        self.audit_mode = os.environ.get('REWARD_REQUIREMENT_AUDIT_MODE', 'joint')
        if self.audit_mode not in {'joint', 'focused', 'hybrid'}:
            raise ValueError('REWARD_REQUIREMENT_AUDIT_MODE must be joint, focused or hybrid')
        self.frame_budget = int(os.environ.get('REWARD_EVIDENCE_FRAMES', '32'))
        if not 8 <= self.frame_budget <= 81:
            raise ValueError('REWARD_EVIDENCE_FRAMES must be between 8 and 81')
        self.contract_registry = None
        self.contract_reviews = {}
        self.contract_registry_sha256 = None
        registry_path = os.environ.get('REWARD_TASK_CONTRACT_REGISTRY')
        if registry_path:
            registry_bytes = Path(registry_path).read_bytes()
            registry = json.loads(registry_bytes)
            if registry.get('schema_version') != 'task-contract-registry-v1':
                raise ValueError('Unsupported task contract registry version')
            self.contract_registry = registry['contracts']
            for key, contract in self.contract_registry.items():
                validate_task_contract(contract)
                if key != contract['source_sha256']:
                    raise ValueError('Task registry key does not match source hash')
            self.contract_reviews = registry.get('contract_reviews', {})
            known_contracts = {c['contract_sha256']: c for c in self.contract_registry.values()}
            for key, review in self.contract_reviews.items():
                if key not in known_contracts or review.get('source_sha256') != known_contracts[key]['source_sha256']:
                    raise ValueError('Contract review does not identify a frozen registry contract')
                if review.get('status') != 'requires_review' or not isinstance(review.get('reason'), str) or not review['reason'].strip():
                    raise ValueError('Contract review must state the unresolved interpretation')
            self.contract_registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
        modules = ('evidence_pipeline.py', 'evidence_video.py', 'evidence_prompts.py', 'evidence_schema.py',
                   'evidence_grounding.py', 'task_contract.py', 'requirement_audit.py', 'focused_audit.py', 'doubao.py', 'process_gates.py', 'process_audit.py', 'physics_completion.py', 'process_events.py', 'training_reward.py')
        digest = hashlib.sha256()
        for name in modules:
            digest.update(name.encode())
            digest.update(Path(__file__).with_name(name).read_bytes())
        digest.update(json.dumps({'model':settings.model,'api_base':settings.api_base,
                                 'frames':self.frame_budget,'max_tokens':self.settings.max_tokens,
                                 'thinking':'disabled','temperature':settings.temperature,
                                 'audit_mode':self.audit_mode,
                                 'gate_policy':self.gate_policy,
                                 'physics_completion_required':self.physics_completion_required,
                                 'task_contract_registry_sha256':self.contract_registry_sha256},sort_keys=True).encode())
        self.evaluator_version = PIPELINE_VERSION + '-' + digest.hexdigest()[:16]
        if physics_hook is not None:
            import inspect
            from scripts.frozen_v41.visibility_evidence import inspect_frame_bytes
            digest.update(inspect.getsource(type(physics_hook)).encode())
            digest.update(inspect.getsource(inspect_frame_bytes).encode())
            digest.update(physics_hook.mode.encode())
            digest.update(json.dumps(physics_hook.request_specs, sort_keys=True).encode())
            self.evaluator_version += '-physics-' + digest.hexdigest()[:16]
        if process_event_hook is not None:
            fingerprint=process_event_hook.fingerprint
            if not isinstance(fingerprint,str) or not re.fullmatch(r'[0-9a-f]{64}',fingerprint):
                raise ValueError('Event hook must fingerprint its implementation and worker configuration')
            self.evaluator_version += '-events-' + fingerprint[:16]

    async def stage(self, name, system_prompt, payload, frames, valid_ids, validator, trace):
        content = [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}] + frames
        base = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}]
        error = None
        for attempt in range(3):
            messages = list(base)
            if error:
                messages.append({'role': 'user', 'content':
                    'The previous output failed structural/evidence validation: ' + error +
                    '. Reassess the supplied source material and emit a corrected full JSON object. '
                    'Do not invent source frame IDs or unsupported facts to satisfy validation.'})
            # Includes exact ordered image bytes and repair text, without persisting images.
            request_sha256 = hashlib.sha256(json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')
            ).encode()).hexdigest()
            try:
                response = await call_llm(messages, self.settings)
            except Exception as exc:
                trace.append({'stage': name, 'attempt': attempt + 1,
                              'messages_sha256': request_sha256,
                              **sampling_trace_fields(getattr(exc, 'sampling_metadata', None)),
                              'error': type(exc).__name__ + ': ' + str(exc)})
                raise
            entry = {'stage': name, 'attempt': attempt + 1, 'response_id': response.get('id'),
                     'messages_sha256': request_sha256,
                     **sampling_trace_fields(response),
                     'usage': response.get('usage'), 'cache_reused': response.get('cache_reused', False),
                     'output': None}
            try:
                text = response['choices'][0]['message']['content']
                parsed = safe_parse_json(text)
                entry['output'] = parsed
                if parsed is None:
                    raise ValueError('Output was not a JSON object')
                validator(parsed, set(valid_ids))
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                error = str(exc)
                entry['validation_error'] = error
                trace.append(entry)
                if response.get('_cache_path'):
                    Path(response['_cache_path']).unlink(missing_ok=True)
                continue
            trace.append(entry)
            return parsed
        trace.append({'stage': name, 'error': 'Evidence validation exhausted: ' + str(error)})
        raise ValueError(f'{name} failed evidence validation after bounded retries: {error}')

    async def prepare_task_contract(self, description, trace):
        def validator(value, _):
            validate_extraction(value, description)

        draft = await self.stage('task_contract_extraction', extraction_prompt(description),
            {'source_task': description}, [], [], validator, trace)
        reviewed = await self.stage('task_contract_coverage_audit', coverage_audit_prompt(description, draft),
            {'source_task': description, 'draft_contract': draft}, [], [], validator, trace)
        return freeze_contract(description, reviewed)

    async def resolve_task_contract(self, description, trace):
        if self.contract_registry is None:
            return await self.prepare_task_contract(description, trace)
        key = hashlib.sha256(description.encode()).hexdigest()
        if key not in self.contract_registry:
            raise ValueError('Task source missing from frozen registry; prepare it before this evaluation')
        contract = self.contract_registry[key]
        validate_task_contract(contract, description)
        return contract

    async def audit_requirements(self, report, contract, stage_name, trace):
        if self.audit_mode == 'joint':
            return await self.stage(stage_name, audit_prompt(contract, report),
                {'instruction': 'Audit only requirement scope and internal textual consistency.'},
                [], [], lambda value, _: validate_requirement_audit(value, contract, report), trace)
        joint = None
        if self.audit_mode == 'hybrid':
            joint = await self.stage(stage_name + '_joint', audit_prompt(contract, report),
                {'instruction': 'Audit only requirement scope and internal textual consistency.'},
                [], [], lambda value, _: validate_requirement_audit(value, contract, report), trace)
            validate_requirement_audit(joint, contract, report)
        semaphore = asyncio.Semaphore(2)
        async def one(requirement):
            requirement_id = requirement['id']
            async with semaphore:
                value = await self.stage(stage_name + '_' + requirement_id,
                    focused_prompt(contract, report, requirement_id),
                    {'instruction': 'Audit only this requirement, even if another has an obvious error.',
                     'target_requirement_id': requirement_id},
                    [], [], lambda value, _: validate_focused_audit(
                        value, contract, report, requirement_id), trace)
            # Validate again at aggregation boundary; never synthesize unchecked success.
            validate_focused_audit(value, contract, report, requirement_id)
            return value['checks'][0]
        checks = await asyncio.gather(*(one(req) for req in contract['requirements']),
                                      return_exceptions=True)
        for check in checks:
            if isinstance(check, BaseException):
                raise check
        result = {'schema_version': 'requirement-scope-audit-v1', 'checks': checks}
        validate_requirement_audit(result, contract, report)
        if joint is not None:
            # Both complete audit outputs remain in the trace. Retain every distinct
            # allegation for image rechecking; agreement is not required or truth.
            for destination, source in zip(result['checks'], joint['checks']):
                for issue in source['issues']:
                    if issue not in destination['issues']:
                        destination['issues'].append(issue)
            validate_requirement_audit(result, contract, report)
        return result

    async def audit_and_repair_requirements(self, report, contract, description,
                                           observations, manifest, frames, coverage, trace):
        """Audit textual scope, repair once against images, and retain unresolved issues.

        The auditor cannot establish visual truth. Its claims are evidence to recheck,
        never instructions to raise a score or an automatic change to a verdict.
        """
        audits = []
        before_repair = report
        ids = [entry['source_frame_index'] for entry in manifest]
        for round_index in range(2):
            audit = await self.audit_requirements(report, contract,
                'requirement_scope_audit' if round_index == 0 else 'requirement_scope_reaudit',
                trace)
            audits.append({
                'report_sha256': hashlib.sha256(json.dumps(
                    report, ensure_ascii=False, sort_keys=True, separators=(',', ':')
                ).encode()).hexdigest(),
                'audit': audit,
            })
            if not any(check['issues'] for check in audit['checks']) or round_index == 1:
                break
            report = await self.stage(
                'verification_scope_repair', verification_prompt(manifest, task_contract=contract),
                {'task_description': description, 'task_contract': contract,
                 'blind_observations': observations, 'draft_report': report,
                 'requirement_scope_audit': audit, 'video_coverage': coverage,
                 'repair_instruction': (
                     'A separate text-only audit found possible requirement-scope or internal '
                     'reasoning errors. It is not visual ground truth. Check each cited claim '
                     'against the original task, its own requirement, and the supplied images. '
                     'Correct only supported errors, retain legitimate failures or uncertainty, '
                     'and do not invent evidence or relax requirements to satisfy the auditor. '
                     'Re-evaluate the complete report consistently; no preferred score or verdict '
                     'is supplied. The frame coverage is unchanged from verification.'),
                 'pipeline_version': MODEL_CONTEXT_VERSION}, frames, ids,
                lambda value, valid_ids: validate_grounded_report(value, valid_ids, contract), trace,
            )
        return report, audits, before_repair

    async def resolve_failure_reward(self, report, contract, scope_audits, contract_review,
                                     physics_evidence, manifest, frames, coverage, trace):
        """Recheck whether task-related doubts can change an otherwise failed reward.

        This conditional decision does not edit the existing assessment or its
        audits. A negative/inconclusive decision preserves review; it never
        instructs the model to produce a preferred score.
        """
        if (contract_review or (physics_evidence or {}).get('input_unobservable')
                or not needs_failure_resolution(report, contract, scope_audits)):
            return None
        candidates = failure_candidates(report, contract, scope_audits)
        ids = [entry['source_frame_index'] for entry in manifest]
        return await self.stage(
            'failure_reward_resolution', failure_resolution_prompt(),
            {'task_contract': contract, 'evidence_report': report,
             'final_scope_audit': scope_audits[-1],
             'candidate_requirement_ids': candidates, 'frame_manifest': manifest,
             'video_coverage': coverage}, frames, ids,
            lambda value, valid_ids: validate_failure_resolution(
                value, report, contract, scope_audits, valid_ids), trace,
        )

    async def process_one_video(self, video_path, description, idx):
        start = time.monotonic()
        trace_root = os.environ.get('REWARD_EVIDENCE_TRACE_DIR')
        identity = hashlib.sha256((str(video_path) + '\n' + description).encode()).hexdigest()[:20]
        trace_path = Path(trace_root) / (identity + '.json') if trace_root else None
        trace = PersistentTrace(trace_path, {'video_path': str(video_path), 'prompt': description,
                                            'pipeline_version': PIPELINE_VERSION, 'evaluator_version': self.evaluator_version})
        source_digest = None
        if self.physics_hook is not None or self.process_event_hook is not None:
            from scripts.frozen_v41.tool_protocol import video_sha256
            source_digest = await asyncio.to_thread(video_sha256, video_path)
        video = await asyncio.to_thread(load_evidence_video, video_path)
        indices = uniform_indices(len(video.frames), self.frame_budget)
        manifest = video.manifest(indices)
        frames = await asyncio.to_thread(video.content, indices)
        # No task text in this stage: desired actions must not masquerade as observations.
        observations = await self.stage('blind_observation', observation_prompt(manifest),
            {'instruction': 'Describe only the supplied visible sequence.', 'pipeline_version': MODEL_CONTEXT_VERSION},
            frames, indices, validate_observations, trace)
        contract = await self.resolve_task_contract(description, trace)

        def report_validator(value, valid_ids):
            validate_grounded_report(value, valid_ids, contract)

        draft = await self.stage('assessment', assessment_prompt(manifest, task_contract=contract),
            {'task_description': description, 'task_contract': contract, 'blind_observations': observations,
             'pipeline_version': MODEL_CONTEXT_VERSION}, frames, indices, report_validator, trace)
        refined = verification_indices(draft, indices, len(video.frames))
        verify_frames = frames if refined == indices else await asyncio.to_thread(video.content, refined)
        coverage = {'decoded_frames': len(video.frames), 'fps': video.fps,
                    'provided_frames': len(refined),
                    'all_decoded_frames_provided': refined == list(range(len(video.frames)))}
        final = await self.stage('verification', verification_prompt(video.manifest(refined), task_contract=contract),
            {'task_description': description, 'task_contract': contract, 'blind_observations': observations, 'draft_report': draft,
             'video_coverage': coverage,
             'coverage_instruction': ('When all_decoded_frames_provided is true, every decoded original frame '
                 'is supplied in temporal order. There are no omitted decoded frames in this verification '
                 'request. Do not explain visible discontinuities using evaluator sampling gaps. '
                 'Original recording frame rate still limits temporal resolution; distinguish visible '
                 'state changes from an unobserved cause.'),
             'pipeline_version': MODEL_CONTEXT_VERSION}, verify_frames, refined, report_validator, trace)
        physics_evidence = None
        if self.physics_hook is not None:
            from scripts.frozen_v41.visibility_evidence import inspect_frame_bytes
            input_visibility = inspect_frame_bytes(
                (frame.shape[1], frame.shape[0], frame.tobytes()) for frame in video.frames)
            if await asyncio.to_thread(video_sha256, video_path) != source_digest:
                raise ValueError('Source video changed after baseline decoding')
            final, physics_evidence = await self.physics_hook.apply(
                pipeline=self, video_path=video_path, video_sha256_expected=source_digest,
                report=final, contract=contract, description=description,
                manifest=video.manifest(refined), frames=verify_frames, valid_ids=refined,
                validator=report_validator, trace=trace, input_visibility=input_visibility)
        final, scope_audits, before_scope_repair = await self.audit_and_repair_requirements(
            final, contract, description, observations, video.manifest(refined),
            verify_frames, coverage, trace,
        )
        scoring = score_report(core_report(final))
        scoring['scoring_version'] = 'evidence-soft-physics-v2-with-scope-review-v5-provisional'
        if physics_evidence and physics_evidence.get('input_unobservable'):
            scoring['total_score'] = None
            scoring['review_reasons'].append('input visibility: all decoded frames are spatially uniform; no grounded manipulation evidence')
            scoring['input_visibility_gate'] = physics_evidence['input_visibility']
        requirement_summary = validate_requirement_checks(final['requirement_checks'], contract, final)
        for requirement_id in requirement_summary['unresolved_requirement_ids']:
            scoring['review_reasons'].append(f'task requirement {requirement_id}: unresolved visible evidence')
        contract_review = self.contract_reviews.get(contract['contract_sha256'])
        if contract_review:
            scoring['review_reasons'].append('task contract: ' + contract_review['reason'])
        for check in scope_audits[-1]['audit']['checks']:
            for issue in check['issues']:
                scoring['review_reasons'].append(
                    f"task requirement {check['requirement_id']}: unresolved scope audit "
                    f"({issue['kind']}): {issue['reason']}"
                )
        scoring['review_required'] = bool(scoring['review_reasons'])
        process_event_evidence = None
        if self.gate_policy == 'evidence_process':
            from reward_as_agent.process_audit import audit_process_with_scope, score_process
            process_ids=refined;process_frames=verify_frames;process_coverage=coverage
            event_context=None;event_reviews=[]
            if self.process_event_hook is not None:
                from reward_as_agent.process_events import validate_event_bundle
                if await asyncio.to_thread(video_sha256,video_path)!=source_digest:
                    raise ValueError('Source changed before event inspection')
                bundle=validate_event_bundle(await self.process_event_hook.apply(
                    pipeline=self,video=video,video_path=video_path,source_sha256=source_digest,
                    description=description,trace=trace),len(video.frames))
                if await asyncio.to_thread(video_sha256,video_path)!=source_digest:
                    raise ValueError('Source changed during event inspection')
                process_ids=sorted(set(refined)|set(bundle['frame_ids']))
                process_frames=await asyncio.to_thread(video.content,process_ids)
                process_coverage={**coverage,'provided_frames':len(process_ids),
                    'all_decoded_frames_provided':process_ids==list(range(len(video.frames)))}
                event_context=bundle['context'];event_reviews=bundle['review_reasons']
                process_event_evidence=bundle['evidence']
                trace.append({'stage':'process_event_evidence','output':bundle})
            process_audit, process_scope_audits = await audit_process_with_scope(self, description, contract,
                video.manifest(process_ids), process_frames, process_ids, process_coverage, trace,
                local_observations=event_context)
            scoring = score_process(scoring, final, process_audit)
            scoring['review_reasons'].extend(event_reviews)
            for check in process_scope_audits[-1]['checks']:
                if check['status'] != 'supported_scope':
                    scoring['review_reasons'].append('process scope unresolved: '+check['reason'])
            scoring['review_required'] = bool(scoring['review_reasons'])
            scoring['reward_gates']['process_scope_audits'] = process_scope_audits
            scoring['reward_gates']['process_frame_manifest'] = video.manifest(process_ids)
            trace.append({'stage': 'reward_gates', **scoring['reward_gates']})
        elif self.gate_policy != 'off':
            from reward_as_agent.process_gates import apply_gates, temporal_evidence
            temporal = (await asyncio.to_thread(temporal_evidence, video.frames)
                        if self.gate_policy == 'process' else None)
            scoring = apply_gates(scoring, final, policy=self.gate_policy, temporal=temporal,
                earlier_reports=[t['output'] for t in trace if t.get('stage') in
                    {'assessment', 'verification', 'physics_tool_reflection'} and
                    isinstance(t.get('output'), dict) and not t.get('validation_error')])
            trace.append({'stage': 'reward_gates', **scoring['reward_gates']})
        if self.physics_completion_required:
            from reward_as_agent.physics_completion import require_physical_completion
            scoring = require_physical_completion(scoring, physics_evidence, source_digest)
            trace.append({'stage': 'physical_completion', **scoring['physical_completion']})
        failure_resolution = await self.resolve_failure_reward(
            final, contract, scope_audits, contract_review, physics_evidence,
            video.manifest(refined), verify_frames, coverage, trace,
        )
        scoring = apply_training_reward(
            scoring, final, contract, scope_audits,
            contract_review=contract_review, physics_evidence=physics_evidence,
            failure_resolution=failure_resolution,
        )
        score = scoring['total_score']
        result = {
            # Kept solely for the existing transport's index/error routing.
            'planning_api_output': {'index': idx, 'score': final['task_assessment']['verdict'], 'status': 'success'},
            'total_score': score if score is not None else -1,
            'pipeline_version': PIPELINE_VERSION, 'evidence_report': final,
            'physics_evidence': physics_evidence,
            'process_event_evidence': process_event_evidence,
            'evaluator_version': self.evaluator_version,
            'task_contract': contract, 'task_contract_sha256': contract['contract_sha256'],
            'task_contract_registry_sha256': self.contract_registry_sha256,
            'task_contract_review': contract_review,
            'requirement_scope_audits': scope_audits,
            'pre_scope_repair_report': before_scope_repair,
            'requirement_summary': requirement_summary,
            'draft_report': draft, 'blind_observations': observations,
            'scoring': scoring, 'review_required': scoring['review_required'],
            'failure_reward_resolution': failure_resolution,
            'diagnostic_score': scoring['diagnostic_score'],
            'diagnostic_review_required': scoring['diagnostic_review_required'],
            'diagnostic_review_reasons': scoring['diagnostic_review_reasons'],
            'training_eligible': score is not None and not scoring['review_required'],
            'frame_manifest': video.manifest(refined),
            'observation_time_manifest': observation_time_manifest(final, video.manifest(refined)),
            'verification_coverage': coverage,
            'video_metadata': {'decoded_frames': len(video.frames), 'fps': video.fps,
                               'width': video.width, 'height': video.height},
            'trace': trace, 'elapsed_seconds': round(time.monotonic() - start, 3),
        }
        if score is None:
            result['error'] = 'needs_review: insufficient observable evidence for numerical reward'
        if trace_root:
            root = Path(trace_root)
            root.mkdir(parents=True, exist_ok=True)
            path = root / (identity + '.json')
            temp = path.with_suffix('.' + str(time.time_ns()) + '.tmp')
            temp.write_text(json.dumps({'video_path': str(video_path), 'prompt': description, **result}, ensure_ascii=False, indent=2))
            temp.replace(path)
        return result
