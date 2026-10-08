"""Prevent local physical evidence from rewriting uninspected process events."""
import copy
import json
from .measurement_support import measurement_support


def key(claim):
    return json.dumps(claim,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def bound_process_revision(before, proposed, physical, expected_sha256):
    issues=[]
    mapping=physical.get('window_mapping',{})
    frames=mapping.get('source_frame_indices',[])
    if physical.get('status')!='ok' or physical.get('video_sha256')!=expected_sha256:
        issues.append('physical result failed or identifies another source')
    valid_frames=isinstance(frames,list) and bool(frames) and all(type(i)is int and i>=0 for i in frames)
    if not valid_frames or frames!=list(range(frames[0],frames[-1]+1)):
        issues.append('physical window does not identify contiguous source frames')
    if mapping.get('source_video_sha256')!=expected_sha256:
        issues.append('physical window source identity mismatch')
    support=None
    entities=physical.get('window_request',{}).get('entities',[])
    if not entities or not {'effector','manipulated_object'}<={e.get('role') for e in entities}:
        issues.append('physical query lacks identifiable hand and manipulated object')
    elif valid_frames:
        try:
            support=measurement_support(physical,entities,frames)
            if not support['continuous_motion_support']:
                issues.append('physical measurements cannot support continuous motion in the inspected window')
        except (ValueError,KeyError,TypeError) as exc:
            issues.append('invalid physical measurements: '+str(exc))
    covered=set(frames) if valid_frames else set()
    old=before['report'];new=proposed['report']
    def negative_claims(report):
        return [c for c in report['claims'] if c['kind']=='execution_error'
                or c['kind']=='continuity' and report['continuity']=='discontinuous'
                or c['kind']=='action' and report['action_evidence']=='absent']
    old_claims={key(c):c for c in negative_claims(old)}
    new_claims={key(c):c for c in negative_claims(new)}
    for encoded,claim in old_claims.items():
        if encoded not in new_claims and not set(claim['frames'])<=covered:
            issues.append('removed or rewritten adverse claim extends outside physical window')
    for encoded,claim in new_claims.items():
        if encoded not in old_claims and not set(claim['frames'])<=covered:
            issues.append('new adverse claim extends outside physical window')
    if old['action_evidence']!=new['action_evidence']:
        old_actions=[c for c in old['claims'] if c['kind']=='action']
        if any(not set(c['frames'])<=covered for c in old_actions):
            issues.append('local evidence cannot invalidate action evidence from another interval')
        if new['action_evidence']=='absent' and frames!=list(range(mapping.get('original_frames',-1))):
            issues.append('a local window cannot establish whole-video absence')
    removed_reviews=set(before['scoring']['review_reasons'])-set(proposed['scoring']['review_reasons'])
    if removed_reviews:issues.append('local process reflection cannot clear existing unbound review reasons')
    if old['quality']!=new['quality'] and not old_claims and not new_claims:
        issues.append('quality changed without an inspectable adverse event')
    if old['quality']!=new['quality'] and any(not set(c['frames'])<=covered for c in list(old_claims.values())+list(new_claims.values())):
        issues.append('overall error severity depends on events outside physical window')
    audit={'version':'physical-event-scope-v2','accepted':not issues,'issues':sorted(set(issues)),
           'measurement_support':support,
           'source_frame_interval':[frames[0],frames[-1]] if frames else None,
           'meaning':'Scope and necessary measurement checks only; passing does not establish that retained or proposed judgments are correct. Rejected measurements are not video defects.'}
    return copy.deepcopy(proposed if not issues else before),audit
