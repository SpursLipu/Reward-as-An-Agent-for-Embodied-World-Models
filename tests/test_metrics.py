import asyncio

from reward_as_agent.config import Settings
from reward_as_agent.metrics import MotionQualityMetrics


def make_settings(enable_motion_quality=True):
    """中文：构造用于动作质量指标测试的最小配置。
English: Build minimal settings for motion-quality metric tests."""
    return Settings(
        api_base="http://localhost:7000/v1",
        api_key="dummy",
        model="model",
        host="0.0.0.0",
        port=7024,
        log_root=__import__("pathlib").Path("runs"),
        save_inputs=True,
        llm_timeout=600,
        max_retries=1,
        motion_quality_path=None,
        enable_motion_quality=enable_motion_quality,
    )


def test_motion_quality_disabled_returns_placeholders():
    """中文：验证动作质量指标当前返回未启用占位结果。
English: Verify that motion-quality metrics currently return disabled placeholders."""
    metrics = MotionQualityMetrics(make_settings(enable_motion_quality=False))

    scores = asyncio.run(metrics.compute(["a.mp4", "b.mp4"]))

    assert scores == [None, None]
