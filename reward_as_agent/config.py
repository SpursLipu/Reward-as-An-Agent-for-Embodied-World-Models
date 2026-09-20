"""Runtime configuration for the Reward as An Agent service."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """中文：配置项格式非法时抛出的异常。
English: Raised when an environment-backed setting has an invalid format."""


def load_dotenv(path: Path = Path(".env")) -> None:
    """中文：从 .env 文件加载尚未设置的环境变量。
English: Load unset environment variables from a .env file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_value(name: str, default: str | None = None) -> str | None:
    """中文：读取环境变量，并在缺失时返回默认值。
English: Read an environment variable, falling back to the default."""
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    """中文：读取布尔型环境变量，并在缺失时返回默认值。
English: Read a boolean environment variable, falling back to the default."""
    value = _env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    """中文：读取整型环境变量，并在格式错误时给出明确异常。
English: Read an integer environment variable and fail clearly on invalid input."""
    value = _env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}.") from exc


def _env_float(name: str, default: float) -> float:
    """中文：读取浮点型环境变量，并在格式错误时给出明确异常。
English: Read a floating-point environment variable and fail clearly on invalid input."""
    value = _env_value(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {value!r}.") from exc


def _env_temperature() -> float | None:
    """Read an optional finite sampling temperature in the inclusive range [0, 2]."""
    name = "REWARD_AS_AGENT_TEMPERATURE"
    value = _env_value(name)
    if value is None or not value.strip():
        return None
    message = f"{name} must be a finite number in [0, 2], got {value!r}."
    try:
        temperature = float(value)
    except ValueError as exc:
        raise ConfigurationError(message) from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ConfigurationError(message)
    return temperature


@dataclass(frozen=True)
class Settings:
    """中文：Reward as An Agent 服务运行时配置集合。
English: Runtime settings used by the Reward as An Agent service."""
    api_base: str
    api_key: str
    model: str
    host: str
    port: int
    log_root: Path
    save_inputs: bool
    llm_timeout: float
    max_retries: int
    dp_size: int
    max_inflight_per_dp: int
    max_tokens: int
    heartbeat_interval: float
    motion_quality_path: str | None
    enable_motion_quality: bool
    provider: str = "openai"
    temperature: float | None = None


def get_settings() -> Settings:
    """中文：汇总环境变量与默认值，构造服务配置对象。
English: Build the service settings from environment variables and defaults."""
    load_dotenv()
    repo_root = Path(__file__).resolve().parents[1]
    return Settings(
        api_base=_env_value("REWARD_AS_AGENT_API_BASE", "http://localhost:7000/v1"),
        api_key=_env_value("REWARD_AS_AGENT_API_KEY", "dummy"),
        model=_env_value("REWARD_AS_AGENT_MODEL", "/path/to/your/reward-model"),
        host=_env_value("REWARD_AS_AGENT_HOST", "0.0.0.0"),
        port=_env_int("REWARD_AS_AGENT_PORT", 7024),
        log_root=Path(_env_value("REWARD_AS_AGENT_LOG_ROOT", str(repo_root / "runs"))),
        save_inputs=_env_bool("REWARD_AS_AGENT_SAVE_INPUTS", True),
        llm_timeout=_env_float("REWARD_AS_AGENT_LLM_TIMEOUT", 600),
        max_retries=_env_int("REWARD_AS_AGENT_MAX_RETRIES", 10),
        dp_size=_env_int("REWARD_AS_AGENT_DP_SIZE", 1),
        max_inflight_per_dp=_env_int("REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP", 8),
        max_tokens=_env_int("REWARD_AS_AGENT_MAX_TOKENS", 8192),
        heartbeat_interval=_env_float("REWARD_AS_AGENT_HEARTBEAT_INTERVAL", 30),
        motion_quality_path=_env_value("REWARD_AS_AGENT_MOTION_QUALITY_PATH") or None,
        enable_motion_quality=_env_bool("REWARD_AS_AGENT_ENABLE_MOTION_QUALITY", False),
        provider=_env_value("REWARD_AS_AGENT_PROVIDER", "doubao"),
        temperature=_env_temperature(),
    )
