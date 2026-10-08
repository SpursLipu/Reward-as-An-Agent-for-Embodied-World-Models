"""Required external evidence is a completion condition, never a quality label."""
import copy


def require_physical_completion(scoring, evidence, source_sha256):
    out=copy.deepcopy(scoring)
    problems=[]
    if not isinstance(source_sha256,str) or not source_sha256:
        problems.append('source identity unavailable')
    records=(evidence or {}).get('tool_records',[])
    if not records:problems.append('no external tool records')
    for index,record in enumerate(records):
        result=record.get('result',{});request=record.get('request',{})
        if result.get('status')!='ok':problems.append(f'tool {index}: {result.get("status","missing status")}')
        elif request.get('video_sha256')!=source_sha256 or result.get('video_sha256')!=source_sha256:
            problems.append(f'tool {index}: source identity mismatch')
    if not (evidence or {}).get('reflection_applied'):problems.append('physical reflection not completed')
    if (evidence or {}).get('fallback_to_baseline'):problems.append('physical evaluation fell back to baseline')
    out['physical_completion']={'required':True,'complete':not problems,'issues':problems,
        'meaning':'Execution and source identity only; successful tools do not establish video quality.'}
    for issue in problems:out['review_reasons'].append('required physical evidence: '+issue)
    out['review_required']=bool(out['review_reasons'])
    return out
