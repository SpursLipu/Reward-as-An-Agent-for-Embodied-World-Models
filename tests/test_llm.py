import pytest

from reward_as_agent.llm import clean_output, safe_parse_json, normalize_model_json


def test_clean_output_strips_json_fence():
    """中文：验证模型输出中的 JSON 代码块包裹会被移除。
English: Verify that JSON code fences are stripped from model output."""
    assert clean_output('```json\n{"x": 1}\n```') == '{"x": 1}'


def test_safe_parse_json_extracts_object():
    """中文：验证解析器能从混合文本中提取 JSON 对象。
English: Verify that the parser extracts a JSON object from mixed text."""
    assert safe_parse_json('prefix {"score": 1} suffix') == {"score": 1}


@pytest.mark.parametrize("content", ["", "not json", "[]"])
def test_safe_parse_json_returns_none_for_non_object(content):
    """中文：验证非对象 JSON 或无效文本会返回 None。
English: Verify that non-object JSON or invalid text returns None."""
    assert safe_parse_json(content) is None


def test_normalize_model_json_removes_explanatory_extras_but_keeps_scores():
    value, removed = normalize_model_json({
        "visual_assessment": {"verdict": "clear", "reason_zh": "清晰", "reason": "clear"},
        "score": 0.5,
        "reason_detail": "extra",
    })
    assert value == {"visual_assessment": {"verdict": "clear", "reason": "clear"}, "score": 0.5}
    assert removed == ["visual_assessment.reason_zh", "reason_detail"]
