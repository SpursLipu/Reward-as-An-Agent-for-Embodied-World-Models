"""Pinned pretrained CoTracker3 inference; no reward or reference-label access."""
from pathlib import Path
import hashlib
import json
import os
import threading
import time
import numpy as np

LOCK = threading.Lock()
MODEL = None
MODEL_META = None


def load_model():
    global MODEL, MODEL_META
    if MODEL is None:
        import torch
        root = Path(os.environ['REWARD_COTRACKER_REPO'])
        checkpoint = Path(os.environ['REWARD_COTRACKER_CHECKPOINT'])
        manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
        for name, expected in manifest['files_sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('CoTracker source integrity mismatch: '+name)
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha != os.environ['REWARD_COTRACKER_SHA256']:
            raise RuntimeError('CoTracker checkpoint integrity mismatch')
        predictor = torch.hub.load(str(root), 'cotracker3_offline', source='local', pretrained=False)
        predictor.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        MODEL = predictor.eval().to('cuda:0')
        MODEL_META = {'model':'CoTracker3 scaled_offline', 'checkpoint_sha256':sha,
                      'source_commit':manifest['commit'], 'device':'cuda:0',
                      'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'visibility':'Official predictor threshold 0.9; predicted visibility is not verified correctness.'}
    return MODEL, MODEL_META


def infer_tracks(video, crop_plan=None):
    import torch
    count=len(video.frames)
    indices=sorted(set(np.linspace(0,count-1,min(count,81)).round().astype(int).tolist()+
                       [c['source_frame_index'] for c in (crop_plan or {}).get('crops',[])]))
    queries=[]; groups=[]
    for y in np.linspace(4,video.height-5,15):
        for x in np.linspace(4,video.width-5,15):
            queries.append([0,float(x),float(y)]);groups.append('full_frame_grid')
    for j,c in enumerate((crop_plan or {}).get('crops',[])):
        t=indices.index(c['source_frame_index'])
        for y in np.linspace(c['y0']*video.height/1000,c['y1']*video.height/1000,7)[1:-1]:
            for x in np.linspace(c['x0']*video.width/1000,c['x1']*video.width/1000,7)[1:-1]:
                queries.append([t,float(x),float(y)]);groups.append('inspection_region_'+str(j))
    source_sha=hashlib.sha256(b''.join(frame.tobytes() for frame in video.frames)).hexdigest()
    key=hashlib.sha256(json.dumps([source_sha,queries,indices,os.environ['REWARD_COTRACKER_SHA256'],
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()],sort_keys=True).encode()).hexdigest()
    cache=Path(os.environ['REWARD_COTRACKER_CACHE'])/(key+'.json')
    with LOCK:
        if cache.exists():return json.loads(cache.read_text())
        model,meta=load_model()
        # OpenCV decoder gives BGR; official CoTracker expects RGB 0..255.
        rgb=np.stack([video.frames[i][:,:,::-1] for i in indices]).copy()
        tensor=torch.from_numpy(rgb).permute(0,3,1,2)[None].float().to('cuda:0')
        q=torch.tensor(queries,dtype=torch.float32,device='cuda:0')[None]
        start=time.monotonic()
        with torch.inference_mode():
            tracks,visible=model(tensor,queries=q,backward_tracking=any(p[0]>0 for p in queries))
        xy=tracks[0].float().cpu().numpy(); vis=visible[0].cpu().numpy().astype(bool)
        del tensor,q,tracks,visible
        result={'model':meta,'source_decoded_sha256':source_sha,'source_indices':indices,
                'queries':queries,'groups':groups,'tracks':xy.round(3).tolist(),
                'visible':vis.tolist(),'elapsed_seconds':round(time.monotonic()-start,3)}
        cache.parent.mkdir(parents=True,exist_ok=True)
        temp=cache.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(cache)
        return result


def segments_from_tracks(raw):
    xy=np.asarray(raw['tracks']);vis=np.asarray(raw['visible'],dtype=bool)
    frames=raw['source_indices'];mid=(len(frames)-1)//2;results=[]
    for a,b in sorted(set([(0,mid),(0,len(frames)-1),(mid,len(frames)-1)])):
        if a==b:continue
        valid=vis[a]&vis[b]&(vis[a:b+1].mean(axis=0)>=.8)&np.isfinite(xy[a:b+1]).all(axis=(0,2))
        tracks=[]
        for i in np.flatnonzero(valid):
            tracks.append({'id':int(i),'group':raw['groups'][i],
                'start_xy':xy[a,i].tolist(),'end_xy':xy[b,i].tolist(),
                'displacement_xy':(xy[b,i]-xy[a,i]).round(2).tolist(),
                'visible_fraction':round(float(vis[a:b+1,i].mean()),3),
                'path_xy':xy[a:b+1,i].tolist(),'path_source_frame_ids':frames[a:b+1]})
        results.append({'start':frames[a],'end':frames[b],'seed_count':xy.shape[1],
            'tracks':tracks,'survival_fraction':round(len(tracks)/xy.shape[1],3),
            'status':'ok' if len(tracks)>=8 else 'insufficient_tracks'})
    return results
