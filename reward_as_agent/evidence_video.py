"""Video evidence with source-frame coordinates, time, and bounded refinement."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import math

from reward_as_agent.video import get_cv2


@dataclass
class EvidenceVideo:
    frames: list
    fps: float
    width: int
    height: int

    def manifest(self, indices):
        return [{"source_frame_index": i, "timestamp_seconds": round(i / self.fps, 4),
                 "view": "full"} for i in indices]

    def content(self, indices):
        cv2 = get_cv2()
        parts = []
        for item in self.manifest(indices):
            i = item['source_frame_index']
            parts.append({"type": "text", "text":
                          f"SOURCE_FRAME {i}; timestamp_seconds={item['timestamp_seconds']}; view=full. "
                          "Reference this source frame number, not the image's position in the message."})
            ok, buf = cv2.imencode('.jpg', self.frames[i], [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f'Could not encode source frame {i}')
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii')}})
        return parts


class InvalidVideoContent(ValueError):
    """Readable input bytes cannot provide valid temporal video evidence."""


def load_evidence_video(path):
    # File access failures are infrastructure errors, not bad generated content.
    with open(path, "rb") as source:
        source.read(1)
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoContent(f'Could not open video: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise InvalidVideoContent('Video has no valid frame rate; temporal evidence cannot be located')
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            raise InvalidVideoContent('Video contains no decoded frames')
        if declared_count > 0 and len(frames) < declared_count - 1:
            raise InvalidVideoContent(f'Video decoding ended prematurely: decoded {len(frames)} of {declared_count} declared frames')
        height, width = frames[0].shape[:2]
        return EvidenceVideo(frames, fps, width, height)
    finally:
        cap.release()


def uniform_indices(total, count=32):
    if total < 1 or count < 2:
        raise ValueError('Need at least one frame and a sampling budget of at least two')
    if total <= count:
        return list(range(total))
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def refinement_indices(report, supplied_indices, total, max_additional=16):
    """Inspect dense adjacent source frames around disputed evidence, never invent time."""
    observations = {o['id']: o for o in report.get('observations', [])}
    priority = []
    for issue in report.get('physics_assessment', {}).get('issues', []):
        for eid in issue.get('evidence', []):
            priority.extend(observations.get(eid, {}).get('frames', []))
    task = report.get('task_assessment', {})
    if task.get('confidence') != 'high' or task.get('target_match') != 'match':
        for eid in task.get('evidence', []):
            priority.extend(observations.get(eid, {}).get('frames', []))
    additions = []
    seen = set(supplied_indices)
    for radius in (1, 2):
        for frame in priority:
            for i in (frame - radius, frame + radius):
                if 0 <= i < total and i not in seen:
                    additions.append(i)
                    seen.add(i)
                    if len(additions) >= max_additional:
                        return sorted(set(supplied_indices) | set(additions))
    return sorted(set(supplied_indices) | set(additions))


def verification_indices(report, supplied_indices, total):
    """Preserve every recorded moment for short clips; bound longer-clip refinement."""
    if total <= 81:
        return list(range(total))
    return refinement_indices(report, supplied_indices, total)
