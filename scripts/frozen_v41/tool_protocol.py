"""Tool execution status is independent of a video's physical validity."""
import hashlib
import json
import re


def video_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_videophy(raw, task):
    if task not in {'pc', 'rule'}:
        raise ValueError('Unsupported task')
    if not isinstance(raw, str):
        raise ValueError('Model output must be text')
    text = raw.strip().lower()
    words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    # Whole-output parsing only: no extraction from explanations or default zeros.
    match = re.fullmatch(r'(zero|one|two|three|four|five|[0-5])[.!]?', text)
    if not match:
        raise ValueError('Ambiguous or noncanonical VideoPhy output')
    value = words.get(match[1], int(match[1]) if match[1].isdigit() else None)
    if value not in ({0, 1, 2} if task == 'rule' else {1, 2, 3, 4, 5}):
        raise ValueError('Label outside task range')
    return {'status': 'abstain' if task == 'rule' and value == 2 else 'ok',
            'raw_score': value if task == 'pc' else None,
            'rule_label': value if task == 'rule' else None,
            'score_direction': 'higher_better' if task == 'pc' else None,
            'abstain_reason': 'rule_not_grounded' if task == 'rule' and value == 2 else None,
            'raw_output': raw}


def evidence_id(request, model_revision, preprocessing_revision):
    # Hash all requested context, including precise frames/interval/entity settings
    # when present. No path-only cache keys or implicit model revisions.
    return hashlib.sha256(json.dumps({'request': request, 'model_revision': model_revision,
        'preprocessing_revision': preprocessing_revision}, sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reflection_evidence(result):
    """Failure and abstention explicitly fall back; never synthesize a reward."""
    if result.get('status') not in {'ok', 'abstain', 'error'}:
        raise ValueError('Unknown tool execution status')
    return {'tool_result': result, 'fallback_to_baseline': result['status'] != 'ok',
            'instruction': ('Treat successful tool output as model evidence, not ground truth. '
                'Do not infer physical correctness from status=ok. Preserve the raw record. '
                'For error/abstain retain baseline judgment and record fallback. '
                'Do not multiply an uncalibrated tool score into the reward.')}
