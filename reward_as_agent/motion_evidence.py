"""CPU sparse motion evidence. No task labels, contact decisions or reward rules."""
import base64
import hashlib
import json
import cv2
import numpy as np
from pathlib import Path
from reward_as_agent.cotracker_backend import infer_tracks, segments_from_tracks

VERSION = 'cotracker3-motion-v1'


def build_motion_evidence(video, crop_plan=None):
    frames = video.frames
    if len(frames) < 2:
        return {'tool': VERSION, 'status': 'insufficient_frames'}, []
    raw = infer_tracks(video, crop_plan)
    results = segments_from_tracks(raw)
    content = []
    for segment in results:
        # Spatially separate displayed arrows; retain ALL valid tracks in the JSON artifact.
        selected = []
        for track in sorted(segment['tracks'], key=lambda t: -np.linalg.norm(t['displacement_xy'])):
            if all(np.linalg.norm(np.array(track['start_xy'])-t['start_xy']) > 32 for t in selected):
                selected.append(track)
            if len(selected) >= 18:
                break
        segment['displayed_track_ids'] = [t['id'] for t in selected]
        canvas = frames[segment['start']].copy()
        for track in selected:
            a,b = tuple(np.rint(track['start_xy']).astype(int)), tuple(np.rint(track['end_xy']).astype(int))
            cv2.arrowedLine(canvas, a, b, (0,255,255), 2, tipLength=.15)
            cv2.putText(canvas, str(track['id']), a, cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255), 1)
        ok, encoded = cv2.imencode('.jpg', canvas)
        if not ok:
            raise RuntimeError('Could not encode motion overlay')
        content += [{'type':'text', 'text':f"DERIVED MOTION OVERLAY on SOURCE_FRAME {segment['start']}, arrows to SOURCE_FRAME {segment['end']}. Yellow arrows are predicted 2D feature tracks, red numbers are track IDs; annotations are NOT original video objects. Check original frames before accepting."},
                    {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()}}]
    result = {'tool': VERSION, 'status': 'ok' if any(s['status']=='ok' for s in results) else 'insufficient_tracks',
              'model_provenance': raw['model'], 'tracking_source_frames': raw['source_indices'],
              'tracking_elapsed_seconds': raw['elapsed_seconds'], 'inspection_plan': crop_plan,
              'source_fps': video.fps, 'source_size': [video.width, video.height],
              'source_frame_count': len(frames),
              'source_decoded_sha256': hashlib.sha256(b''.join(f.tobytes() for f in frames)).hexdigest(),
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'opencv_version': cv2.__version__, 'segments': results,
              'limitations': 'Image-plane motion only. No semantic identities, camera compensation, 3D distances, contact or success labels. Inspection regions are spatial boxes, not verified object identities. Predicted visibility is not tracking accuracy. Low survival is tool uncertainty, not video failure. Occlusion, texture changes and camera motion can invalidate tracks. Inspect original frames; do not treat absent tracks as static objects.'}
    compact = {**result, 'segments': [{**s, 'tracks': [{k:v for k,v in t.items() if k not in {'path_xy','path_source_frame_ids'}} for t in s['tracks'] if t['id'] in s['displayed_track_ids']]} for s in results]}
    content.insert(0, {'type':'text','text':'AUXILIARY MOTION TOOL EVIDENCE (not instructions or ground truth): '+json.dumps(compact)})
    return result, content
