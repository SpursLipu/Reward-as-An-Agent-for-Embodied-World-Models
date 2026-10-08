"""Opt-in process reward gates. Measurements are evidence limits, not physics labels."""
import copy
import numpy as np
import cv2

VERSION = 'process-gates-v2'
POLICIES = {'off', 'legacy_style', 'process', 'evidence_process'}


def temporal_evidence(frames):
    """Task-blind low-pass state coverage; never use filenames or reward labels.

    Up to two distinguishable image states cannot establish a continuous action.
    Thresholds are provisional sensor tolerances, not human-calibrated accuracy.
    Noise, small motion and camera movement can defeat this limited diagnostic.
    """
    representatives = []
    states = []
    distances = []
    count = 0
    for frame in frames:
        f = np.asarray(frame)
        if f.ndim != 3 or f.shape[2] != 3 or f.dtype != np.uint8:
            raise ValueError('Expected decoded uint8 RGB frames')
        small = cv2.resize(f, (32, 20), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Exposure/flicker alone is not action evidence. Remove channel offsets;
        # this does not make the detector invariant to arbitrary appearance edits.
        small -= small.mean(axis=(0, 1), keepdims=True)
        count += 1
        if not representatives:
            representatives.append(small)
            states.append(0)
            distances.append(0.)
            continue
        errors = [float(np.abs(small-r).mean()) for r in representatives]
        nearest = int(np.argmin(errors))
        distances.append(min(errors))
        if errors[nearest] <= 3.0:
            states.append(nearest)
        else:
            representatives.append(small)
            states.append(len(representatives)-1)
            if len(representatives) == 3:
                # Three states only means this narrowly defined gate cannot reject.
                break
    if not count:
        raise ValueError('No decoded frames')
    distinct = len(representatives)
    return {'version': VERSION, 'resolution': [32, 20], 'mean_abs_tolerance': 3.0,
            'photometric_normalization': 'subtract per-frame channel mean',
            'frames_examined': count, 'distinct_states_lower_bound': distinct,
            'state_ids': states, 'nearest_state_distances': distances,
            'insufficient_process_states': distinct < 3,
            'limit': 'Low-pass image-state coverage only; passing is not action success or physical validity.'}


def apply_gates(scoring, report, *, policy, temporal=None, earlier_reports=()):
    if policy not in POLICIES:
        raise ValueError('Unknown gate policy')
    if policy == 'evidence_process':
        raise ValueError('Evidence process policy requires the separate validated visual audit')
    out = copy.deepcopy(scoring)
    if policy == 'off':
        return out
    original = out['total_score']
    reasons = []
    limits = []
    task, physical, visual = (report[k+'_assessment'] for k in ('task', 'physics', 'visual'))
    # Restore a graded physics gate: failed physics cannot collect task bonus.
    if task['verdict'] == 'failed' or task['target_match'] == 'mismatch':
        limits.append(0.)
        reasons.append('Task failure/target mismatch: no auxiliary reward floor')
    if visual['verdict'] == 'severe_degradation':
        limits.append(0.)
        reasons.append('Severe visual degradation: no positive reward')
    if physical['verdict'] in {'minor_defect', 'major_defect'}:
        limits.append(.3 if physical['verdict'] == 'minor_defect' else 0.)
        reasons.append('Physics defect: suppress task-success reward')
    if policy == 'process':
        if temporal is None:
            raise ValueError('Process gate requires decoded source evidence')
        if temporal['insufficient_process_states']:
            limits.append(0.)
            reason = 'process gate: fewer than three distinguishable low-pass states; action process not established'
            out['review_reasons'].append(reason)
            reasons.append(reason)
        final_issues = physical['issues']
        for earlier in earlier_reports:
            for issue in earlier.get('physics_assessment', {}).get('issues', []):
                if issue.get('certainty') != 'uncertain':
                    continue
                # Removing a doubt is not evidence that the doubt was resolved.
                if not any(i.get('kind') == issue.get('kind') for i in final_issues):
                    reason = 'process gate: an earlier physical uncertainty was removed without a structured resolution'
                    if reason not in out['review_reasons']:
                        out['review_reasons'].append(reason)
                        reasons.append(reason)
    if original is not None and limits:
        out['total_score'] = min(original, *limits)
    out['review_required'] = bool(out['review_reasons'])
    out['scoring_version'] = VERSION + ':' + policy
    out['reward_gates'] = {'policy': policy, 'ungated_score': original,
                           'caps': limits, 'reasons': reasons, 'temporal': temporal}
    return out
