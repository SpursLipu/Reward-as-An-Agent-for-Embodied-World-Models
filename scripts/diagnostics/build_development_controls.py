"""Explicit negative controls, separate from real videos and human labels."""
import argparse
import json
from pathlib import Path
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--cases',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
cases=json.loads(args.cases.read_text())
manifest=[]
for case in cases:
    manifest.append({k:case[k] for k in ('sample_id','video_path','prompt','old_score','new_score','audit_id')})
for case in (cases[0],cases[3]):
    cap=cv2.VideoCapture(case['video_path'])
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok,first=cap.read()
    cap.release()
    if not ok or fps<=0 or count<2:
        raise ValueError('Invalid source for negative control')
    output=args.output/(case['audit_id']+'_frozen_first.mp4')
    if output.exists():
        raise ValueError('Refuse to overwrite frozen control')
    h,w=first.shape[:2]
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():
        raise ValueError('Video writer could not open')
    for _ in range(count):writer.write(first)
    writer.release()
    manifest.append({'sample_id':'control/'+case['audit_id']+'_frozen_first', 'video_path':str(output),
                     'prompt':case['prompt'],'synthetic_control':True,'control_source_id':case['sample_id'],
                     'expected_task_verdict':'failed','expectation_basis':'All frames repeat initial state before required action; synthetic control, not human rating.'})
case=cases[3]
manifest.append({'sample_id':'control/case_04_wrong_target','video_path':case['video_path'],
                 'prompt':'The right robot arm picks up a blue cube and places the blue cube inside the white bin.',
                 'synthetic_control':True,'control_source_id':case['sample_id'],
                 'expected_task_verdict':'failed','expectation_basis':'The source shows the left arm lifting a textured round object; target and action are deliberately changed. Not a human rating.'})
(args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'real_development_cases':len(cases),'synthetic_controls':len(manifest)-len(cases)}))
