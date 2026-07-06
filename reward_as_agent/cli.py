"""Command line interface for Reward as An Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import uvicorn

from reward_as_agent import __version__
from reward_as_agent.config import get_settings


def serve(_: argparse.Namespace) -> None:
    """中文：按当前配置启动 FastAPI reward 服务。
English: Start the FastAPI reward service with the current settings."""
    settings = get_settings()
    uvicorn.run(
        "reward_as_agent.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


def evaluate(args: argparse.Namespace) -> None:
    """中文：向正在运行的服务提交视频评估请求并流式打印结果。
English: Submit videos to a running service and stream evaluation results."""
    payload = {
        "video_path": [str(Path(path).expanduser().resolve()) for path in args.video],
        "prompt": args.prompt,
    }
    response = requests.post(args.url, json=payload, stream=True, timeout=args.timeout)
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        try:
            print(json.dumps(json.loads(line), ensure_ascii=False), flush=True)
        except json.JSONDecodeError:
            print(line.decode("utf-8", errors="replace"), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """中文：构造 reward-as-agent 命令行参数解析器。
English: Build the argument parser for the reward-as-agent command."""
    parser = argparse.ArgumentParser(
        prog="reward-as-agent",
        description=(
            "Reward as An Agent for Embodied World Models: "
            "agentic reward evaluation service for embodied world-model videos."
        ),
    )
    parser.add_argument("--version", action="version", version=f"reward-as-agent {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="start the FastAPI reward service")
    serve_parser.set_defaults(func=serve)

    eval_parser = subparsers.add_parser("eval", help="send videos to a running reward service")
    eval_parser.add_argument("--url", default="http://127.0.0.1:7024/eval_video")
    eval_parser.add_argument("--prompt", required=True, help="task prompt/description shared by the videos")
    eval_parser.add_argument("--video", action="append", required=True, help="video path; repeat for batches")
    eval_parser.add_argument("--timeout", type=float, default=600)
    eval_parser.set_defaults(func=evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    """中文：解析命令行参数并分发到对应子命令。
English: Parse command-line arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
