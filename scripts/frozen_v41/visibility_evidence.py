"""Exact spatial-uniformity evidence, not a general visibility or quality model."""


def inspect_frame_bytes(frames):
    count = uniform = 0
    for width, height, raw in frames:
        if width < 1 or height < 1 or not isinstance(raw, bytes) or len(raw) != width*height*3:
            raise ValueError('Expected packed uint8 three-channel image bytes')
        count += 1
        uniform += raw == raw[:3]*(width*height)
    if not count:
        raise ValueError('No decoded frames')
    return {'method': 'exact-spatially-uniform-rgb-v1', 'decoded_frames': count,
            'spatially_uniform_frames': uniform, 'all_frames_spatially_uniform': uniform == count,
            'limit': 'Only detects exactly uniform decoded frames; does not judge general visibility, task completion, or physics.'}
