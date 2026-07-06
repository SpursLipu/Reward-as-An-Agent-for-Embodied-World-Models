from reward_as_agent.video import build_content, build_physics_reflection_content


def test_build_content_handles_missing_frames():
    """中文：验证缺失帧时仍能构造纯文本多模态输入。
English: Verify that text-only multimodal content is built when frames are missing."""
    assert build_content(None, "task") == [{"type": "text", "text": "task"}]


def test_build_physics_reflection_content_serializes_inputs():
    """中文：验证物理反思输入会序列化已有结果并标记缺失结果。
English: Verify that physics reflection content serializes present results and marks missing ones."""
    content = build_physics_reflection_content(["abc"], {"x": 1}, None, {"y": 2})

    assert content[0]["type"] == "text"
    assert '"x": 1' in content[0]["text"]
    assert "计算失败无法输入" in content[0]["text"]
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc"
