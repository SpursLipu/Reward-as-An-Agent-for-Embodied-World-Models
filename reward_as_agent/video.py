"""Video decoding and frame formatting utilities."""

from __future__ import annotations

import base64


def get_cv2():
    """中文：延迟导入 OpenCV，并在缺失依赖时给出安装提示。
English: Lazily import OpenCV and provide an install hint when it is missing."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video decoding. Install the project dependencies with "
            "`python -m pip install -e .` or `bash scripts/setup_env.sh`."
        ) from exc
    return cv2


def extract_frames(video_path, frame_count=16, shift=0):
    """中文：从视频首尾和中间均匀抽帧，并编码为 JPEG base64。
English: Sample first, last, and uniform middle frames as base64 JPEG images."""
    cv2 = get_cv2()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first_frame_idx = 0
    last_frame_idx = total_frames - 1
    middle_need = frame_count - 2

    sample_indices = [first_frame_idx]

    if middle_need > 0 and total_frames > 2:
        step = (last_frame_idx - first_frame_idx) / (middle_need + 1)
        for i in range(1, middle_need + 1):
            idx = int(first_frame_idx + step * i + shift)
            idx = max(first_frame_idx + 1, min(idx, last_frame_idx - 1))
            sample_indices.append(idx)

    sample_indices.append(last_frame_idx)

    frames_b64 = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frames_b64.append(base64.b64encode(buf).decode())

    cap.release()
    return frames_b64


def extract_all_frames(video_path, jpeg_quality=70):
    """中文：解码视频所有帧，同时返回 RGB 数组和 OpenAI 图像输入格式。
English: Decode every frame and return both RGB arrays and OpenAI image payloads."""
    raw_rgb_frames = []
    openai_frames = []

    cv2 = get_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_rgb_frames.append(frame_rgb)

        encode_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", encode_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        openai_frames.append(base64.b64encode(buf).decode("utf-8"))

    cap.release()
    if not openai_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return raw_rgb_frames, openai_frames


def sample_uniform_frames(frames, sample_count):
    """中文：在保留首尾帧的前提下，对帧序列做均匀采样。
English: Uniformly sample a frame sequence while preserving the first and last frames."""
    total_frames = len(frames)

    if total_frames == 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [frames[0]]
    if sample_count >= total_frames:
        return frames

    sampled = [frames[0]]
    if sample_count == 2:
        sampled.append(frames[-1])
        return sampled

    need_middle = sample_count - 2
    step = (total_frames - 1) / (need_middle + 1)
    for i in range(1, need_middle + 1):
        idx = int(round(step * i))
        sampled.append(frames[idx])

    sampled.append(frames[-1])
    return sampled


def build_content(frames_b64=None, description=None):
    """中文：构造多模态模型请求中的文本与图像 content 列表。
English: Build the text-and-image content list for a multimodal model request."""
    content = [{"type": "text", "text": description or "请执行任务！"}]
    for b64 in frames_b64 or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return content


def safe_json_check(x):
    """中文：为反思阶段补齐缺失的 JSON 结果占位说明。
English: Provide a placeholder message when a reflection input JSON is missing."""
    fail_msg = "计算失败无法输入，需要在reflection阶段重新给出精确的数值"
    if x is None:
        return fail_msg
    return x


def build_physics_reflection_content(
    frames_b64,
    json_output,
    interpenetration_json_output,
    shape_json_output,
):
    """中文：组合物理相关评估结果与视频帧，生成反思阶段输入。
English: Combine physics-related judgments and frames into reflection-stage input."""
    import json

    json_output = json.dumps(safe_json_check(json_output), ensure_ascii=False)
    interpenetration_json_output = json.dumps(
        safe_json_check(interpenetration_json_output), ensure_ascii=False
    )
    shape_json_output = json.dumps(safe_json_check(shape_json_output), ensure_ascii=False)

    content = [
        {
            "type": "text",
            "text": json_output + "\n" + interpenetration_json_output + "\n" + shape_json_output,
        }
    ]
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    return content
