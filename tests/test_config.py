from pathlib import Path

import pytest

from reward_as_agent.config import ConfigurationError, get_settings, load_dotenv


def test_load_dotenv_preserves_existing_env(monkeypatch, tmp_path: Path):
    """中文：验证 .env 加载不会覆盖已存在的环境变量。
English: Verify that .env loading does not overwrite existing environment variables."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REWARD_AS_AGENT_MODEL=from-file\nREWARD_AS_AGENT_PORT=9000\n# ignored\nBAD_LINE\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REWARD_AS_AGENT_MODEL", "from-env")

    load_dotenv(env_file)

    assert __import__("os").environ["REWARD_AS_AGENT_MODEL"] == "from-env"
    assert __import__("os").environ["REWARD_AS_AGENT_PORT"] == "9000"


def test_get_settings_rejects_invalid_port(monkeypatch):
    """中文：验证非法端口配置会触发配置异常。
English: Verify that an invalid port setting raises a configuration error."""
    monkeypatch.setenv("REWARD_AS_AGENT_PORT", "not-a-port")

    with pytest.raises(ConfigurationError, match="REWARD_AS_AGENT_PORT"):
        get_settings()

