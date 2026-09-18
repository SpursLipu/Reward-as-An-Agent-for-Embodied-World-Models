"""Build two explicit diagnostic corruptions, never human or natural-defect labels.

The temporal control alternates source first/last states at every position in the
evaluator's 32-frame uniform sample. Short holds preserve detectable state changes
both in that sample and in adjacent source frames. Re-decoding verifies this claim.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Source requires decoded frames and a valid FPS")
        if reported_count > 0 and reported_count != len(frames):
            raise ValueError("Decoded frame count differs from container count")
        if any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("Changing frame dimensions are not supported")
        return frames, fps
    finally:
        capture.release()


def indices(total, count=32):
    if total < count:
        raise ValueError("Control requires at least 32 source frames")
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def mad(left, right):
    return float(np.mean(cv2.absdiff(left, right)))


def write_video(path, frames, fps):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise ValueError("MP4 writer could not open")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def build(cases_path, output):
    cases = json.loads(cases_path.read_text())
    candidates = [case for case in cases if case.get("audit_id") == "case_04"
                  and not case.get("synthetic_control")]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one real case_04 source in cases manifest")
    case = candidates[0]
    source = Path(case["video_path"])
    source_hash = sha256(source)
    frames, fps = decode(source)
    count = len(frames)
    sample_indices = indices(count)
    source_delta = mad(frames[0], frames[-1])
    changed_fraction = float(np.mean(np.any(cv2.absdiff(frames[0], frames[-1]) > 16, axis=2)))
    if source_delta < 2.0 or changed_fraction < 0.02:
        raise ValueError("First/last source states insufficiently distinct for this diagnostic")
    # output.mkdir is deliberately exclusive. Never reuse or overwrite an old control set.
    output.mkdir(parents=False, exist_ok=False)
    state_for_frame = [(bisect.bisect_right(sample_indices, i) - 1) % 2 for i in range(count)]
    state_frames = [frames[0], frames[-1]]
    temporal = [state_frames[state] for state in state_for_frame]
    black_frame = np.zeros_like(frames[0])
    paths = {"temporal": output / "case_04_repeated_state_jumps.mp4",
             "black": output / "case_04_all_black.mp4"}
    write_video(paths["temporal"], temporal, fps)
    write_video(paths["black"], [black_frame] * count, fps)
    decoded = {}
    for name, path in paths.items():
        recovered, recovered_fps = decode(path)
        if len(recovered) != count or recovered[0].shape != frames[0].shape:
            raise AssertionError(f"{name}: frame count or resolution changed")
        if abs(recovered_fps - fps) > 0.01:
            raise AssertionError(f"{name}: FPS changed")
        decoded[name] = recovered

    sampled_deltas = [mad(decoded["temporal"][a], decoded["temporal"][b])
                      for a, b in zip(sample_indices, sample_indices[1:])]
    boundary_deltas = [mad(decoded["temporal"][i - 1], decoded["temporal"][i])
                       for i in sample_indices[1:]]
    matched_states = []
    for i in sample_indices:
        distances = [mad(decoded["temporal"][i], state) for state in state_frames]
        matched_states.append(int(np.argmin(distances)))
    expected_states = [i % 2 for i in range(len(sample_indices))]
    if matched_states != expected_states:
        raise AssertionError("Encoded states no longer alternate in the 32-frame evaluator sample")
    if min(sampled_deltas + boundary_deltas) < source_delta * 0.4:
        raise AssertionError("A sampled transition or adjacent-frame jump was lost during encoding")
    black_mean = float(np.mean([np.mean(frame) for frame in decoded["black"]]))
    black_max = max(int(frame.max()) for frame in decoded["black"])
    if black_mean > 1.0 or black_max > 2:
        raise AssertionError("Encoded black video contains unexpected visible pixels")
    if sha256(source) != source_hash:
        raise AssertionError("Original source bytes changed")

    common = {"synthetic_control": True, "diagnostic_only": True, "human_label": False,
              "control_source_id": case["sample_id"], "prompt": case["prompt"]}
    manifest = [
        {"sample_id": case["sample_id"], "video_path": str(source), "prompt": case["prompt"],
         "role": "unchanged_same_task_reference", "synthetic_control": False,
         "source_sha256": source_hash},
        {**common, "sample_id": "control/case_04_repeated_state_jumps",
         "video_path": str(paths["temporal"]),
         "construction": "Alternate source first/last states in short holds, switching at each 32-frame uniform sample index.",
         "expected_observation": "Detect repeated temporal/state discontinuities, or explicitly request review when interpretation is uncertain.",
         "claim_limit": "Artificial editing discontinuity; not a naturally occurring physics defect, human label, or uniquely prescribed physics category."},
        {**common, "sample_id": "control/case_04_all_black", "video_path": str(paths["black"]),
         "construction": "Replace every original frame with zeros while preserving frame count, FPS and resolution.",
         "expected_observation": "Recognize the scene/task as visually unobservable, or require review; no visual basis exists to confirm task completion.",
         "claim_limit": "Synthetic removal of visible evidence; no numeric human score or uniquely prescribed physics category."},
    ]
    metadata = {
        "schema_version": "visibility-controls-v1", "human_annotations": 0,
        "source_sample_id": case["sample_id"], "source_video": str(source), "source_sha256": source_hash,
        "source_unchanged_verified": True, "codec": "mp4v", "fps": fps, "decoded_frame_count": count,
        "width": frames[0].shape[1], "height": frames[0].shape[0],
        "source_state_frame_indices": [0, count - 1], "state_for_each_output_frame": state_for_frame,
        "uniform_sampling_count": 32, "uniform_sample_indices": sample_indices,
        "source_state_pixel_MAD_0_255": source_delta,
        "source_state_fraction_pixels_changed_over_16": changed_fraction,
        "decoded_sampled_state_assignment": matched_states,
        "decoded_sampled_transition_MAD_0_255": sampled_deltas,
        "decoded_adjacent_boundary_MAD_0_255": boundary_deltas,
        "black_decoded_pixel_mean_0_255": black_mean, "black_decoded_pixel_max_0_255": black_max,
        "control_sha256": {name: sha256(path) for name, path in paths.items()},
        "verification": "Re-decoded both videos; verified resolution/FPS/count, all 31 sampled state changes and adjacent-frame boundaries, black pixel statistics, and unchanged source hash.",
        "interpretation_limit": "Construction establishes diagnostic input properties, not natural-defect accuracy or human agreement. Do not mix these controls into real-video score distributions or human-label files.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "construction.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "synthetic_controls": 2, "unchanged_references": 1,
                      "decoded_frames": count, "fps": fps, "source_state_MAD": source_delta,
                      "minimum_sampled_transition_MAD": min(sampled_deltas),
                      "minimum_adjacent_boundary_MAD": min(boundary_deltas),
                      "black_mean": black_mean, "black_max": black_max,
                      "source_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.cases, args.output)
