"""Repeated text-only probe; synthetic expectations stay outside model input."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from reward_as_agent.config import get_settings
from reward_as_agent.doubao import call_doubao
from reward_as_agent.focused_audit import focused_prompt, validate_focused_audit
from reward_as_agent.llm import safe_parse_json
from reward_as_agent.requirement_audit import audit_prompt, validate_requirement_audit


async def run(fixtures, output, mode):
    output.mkdir(parents=True, exist_ok=False)
    raw=fixtures.read_bytes()
    (output/'fixtures.json').write_bytes(raw)
    (output/'config.json').write_text(json.dumps({'fixtures_sha256':hashlib.sha256(raw).hexdigest(),
        'repeats':3,'temperature':0,'mode':mode,'human_accuracy_established':False},indent=2))
    items=json.loads(raw)
    settings=get_settings()
    if settings.temperature != 0:
        raise ValueError('Probe requires temperature zero')
    semaphore=asyncio.Semaphore(2)
    async def one(item,repeat):
        prompt=focused_prompt(item['contract'],item['report'],item['target'])
        record={'name':item['name'],'repeat':repeat,'target':item['target'],
            'expect_issue':item['expect_issue'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        async with semaphore:
            try:
                response=await call_doubao([{'role':'user','content':prompt}],settings)
                record['response']=response
                audit=safe_parse_json(response['choices'][0]['message']['content'])
                validate_focused_audit(audit,item['contract'],item['report'],item['target'])
                record['focused_audit'] = audit
                if mode == 'hybrid':
                    joint_prompt = audit_prompt(item['contract'], item['report'])
                    joint_response = await call_doubao([{'role':'user','content':joint_prompt}],settings)
                    record['joint_response'] = joint_response
                    record['joint_prompt_sha256'] = hashlib.sha256(joint_prompt.encode()).hexdigest()
                    joint = safe_parse_json(joint_response['choices'][0]['message']['content'])
                    validate_requirement_audit(joint,item['contract'],item['report'])
                    record['joint_audit'] = joint
                    # Copy the selected check so the recorded independent audit is immutable.
                    audit = json.loads(json.dumps(audit))
                    selected = next(c for c in joint['checks'] if c['requirement_id']==item['target'])
                    for issue in selected['issues']:
                        if issue not in audit['checks'][0]['issues']:
                            audit['checks'][0]['issues'].append(issue)
                record.update(status='success',audit=audit,
                    target_check_pass=(bool(audit['checks'][0]['issues'])==item['expect_issue']
                                       if item['expect_issue'] is not None else None))
            except Exception as exc:
                record.update(status='error',error_type=type(exc).__name__)
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ['name','repeat','status','target_check_pass']}),flush=True)
    await asyncio.gather(*(one(item,repeat) for repeat in range(3) for item in items))
    records=[json.loads(x) for x in (output/'results.jsonl').read_text().splitlines()]
    code=0 if all(r['status']=='success' for r in records) else 1
    (output/'exit_status').write_text(str(code)+'\n')
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['focused','hybrid'],default='focused')
    args=parser.parse_args()
    raise SystemExit(asyncio.run(run(args.fixtures,args.output,args.mode)))
