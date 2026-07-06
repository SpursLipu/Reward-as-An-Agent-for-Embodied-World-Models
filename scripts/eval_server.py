"""Small client for smoke-testing a running Reward as An Agent server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    """中文：解析烟测客户端的服务地址、prompt 和视频路径参数。
English: Parse server URL, prompt, and video paths for the smoke-test client."""
    parser = argparse.ArgumentParser(description="Send videos to a Reward as An Agent server.")
    parser.add_argument("--url", default="http://127.0.0.1:7024/eval_video")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video", action="append", required=True, help="repeat for multiple videos")
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def main() -> None:
    """中文：向 reward 服务发送请求，并打印流式 JSONL 响应。
English: Send a request to the reward service and print streamed JSONL responses."""
    args = parse_args()
    payload = {
        "video_path": [str(Path(path).expanduser().resolve()) for path in args.video],
        "prompt": args.prompt,
    }
    response = requests.post(args.url, json=payload, stream=True, timeout=args.timeout)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            print(json.dumps(json.loads(line), ensure_ascii=False))


if __name__ == "__main__":
    main()
