"""Validate the event hook boundary before its evidence reaches scoring."""


def validate_event_bundle(bundle,total):
    if not isinstance(bundle,dict) or set(bundle)!={'context','frame_ids','evidence','review_reasons'}:
        raise ValueError('Event hook must return evidence, not scores')
    ids=bundle['frame_ids']
    if not isinstance(ids,list) or not ids or any(type(i)is not int or not 0<=i<total for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('Event hook returned invalid source frames')
    if bundle['context'] is not None and not isinstance(bundle['context'],dict):raise ValueError('Invalid event context')
    if not isinstance(bundle['evidence'],dict):raise ValueError('Invalid event evidence record')
    reasons=bundle['review_reasons']
    if not isinstance(reasons,list) or any(not isinstance(r,str) or not r.strip() for r in reasons):
        raise ValueError('Invalid event review reasons')
    return bundle
