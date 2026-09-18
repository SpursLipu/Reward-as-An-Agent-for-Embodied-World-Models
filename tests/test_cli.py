import pytest

from reward_as_agent import __version__
from reward_as_agent.cli import main


def test_cli_version(capsys):
    """中文：验证 CLI 能正确打印当前包版本。
English: Verify that the CLI prints the current package version."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert f"reward-as-agent {__version__}" in capsys.readouterr().out
