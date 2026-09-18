"""Disabled motion-quality metric placeholder."""

from __future__ import annotations


class MotionQualityMetrics:
    """中文：动作质量指标占位适配器，当前开源版本不执行实际计算。
English: Placeholder adapter for motion-quality metrics; the open release does not compute them."""

    def __init__(self, settings):
        """中文：保存配置并显式标记动作质量指标未启用。
English: Store settings and explicitly mark motion-quality metrics as unavailable."""
        self.settings = settings
        self.import_error = None
        self.available = False

    async def compute(self, video_paths: list[str]):
        """中文：返回动作质量占位结果，表示该模块当前未纳入总分。
English: Return placeholder motion-quality results to show the module is excluded."""
        return [None] * len(video_paths)
