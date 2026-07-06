import asyncio
from pathlib import Path

from reward_as_agent.logging import save_video_and_prompt


def test_save_video_and_prompt_without_system_tzdata(tmp_path: Path):
    """中文：验证日志目录创建不依赖系统 tzdata。
English: Verify log directory creation does not depend on system tzdata."""
    video = tmp_path / "input.mp4"
    video.write_bytes(b"demo")

    folder = asyncio.run(save_video_and_prompt([str(video)], "task prompt", str(tmp_path / "runs")))

    output = Path(folder)
    assert (output / "prompt.txt").read_text(encoding="utf-8") == "task prompt"
    assert (output / "video_0.mp4").read_bytes() == b"demo"
