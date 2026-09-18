"""Export original-resolution video frames for visual audit; never generate labels."""
import argparse
import json
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--sample-prefix', required=True)
parser.add_argument('--frames', default='52,65,80')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
rows = json.loads(args.manifest.read_text())
if isinstance(rows, dict):
    rows = rows['samples']
requested = sorted(set(int(i) for i in args.frames.split(',')))
if not requested or requested[0] < 0:
    parser.error('frames must contain nonnegative source-frame indices')
args.output.mkdir(parents=True, exist_ok=True)
records = []
for row in rows:
    if not row['sample_id'].startswith(args.sample_prefix):
        continue
    cap = cv2.VideoCapture(row['video_path'])
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open ' + row['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        index = 0
        found = set()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in requested:
                name = row['sample_id'].replace('/', '_') + f'_frame_{index}.jpg'
                destination = args.output / name
                if destination.exists():
                    raise FileExistsError(destination)
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                    raise ValueError('Image encoding failed')
                found.add(index)
                records.append({'sample_id': row['sample_id'], 'video_path': row['video_path'],
                                'source_frame_index': index, 'timestamp_seconds': index / fps,
                                'image': str(destination), 'resized': False})
            index += 1
        if found != set(requested):
            raise ValueError('Some requested frames were not decoded')
    finally:
        cap.release()
if not records:
    raise ValueError('No matching samples')
(args.output / 'frames.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
print(json.dumps({'exported_frames': len(records)}))
