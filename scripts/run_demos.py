#!/usr/bin/env python3
"""Run the bundled demos against the current Agent and retain auditable results."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_demos(repo_root: Path | None = None) -> tuple[str, ...]:
    """Find bundled demo inputs; payload and video validation stays in load_case."""
    examples = (repo_root or REPO_ROOT) / "examples"
    if not examples.is_dir():
        return ()
    return tuple(sorted(
        directory.name for directory in examples.iterdir()
        if re.fullmatch(r"demo_[0-9]{2}", directory.name)
        and directory.is_dir()
        and (directory / "request.json").is_file()
        and (directory / "prompt.txt").is_file()
    ))


# Keep the shared import used by the standalone WMReward demo runner.
DEMOS = discover_demos()
PUBLIC_HEALTH_FIELDS = (
    "status", "model", "provider", "pipeline", "dp_size",
    "max_inflight_per_dp", "max_tokens", "heartbeat_interval",
    "motion_quality_enabled", "motion_quality_available",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def service_urls(value: str) -> tuple[str, str]:
    parts = urlsplit(value.rstrip("/"))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("--url must be an HTTP(S) service URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("--url must not contain credentials, a query, or a fragment")
    base_path = parts.path.removesuffix("/eval_video").rstrip("/")
    base = urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))
    return base + "/health", base + "/eval_video"


def make_session(url: str) -> requests.Session:
    session = requests.Session()
    hostname = urlsplit(url).hostname
    try:
        loopback = ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        loopback = hostname == "localhost"
    if loopback:
        session.trust_env = False
    return session


def load_case(demo: str) -> tuple[dict, list[Path]]:
    directory = REPO_ROOT / "examples" / demo
    request_path = directory / "request.json"
    prompt_path = directory / "prompt.txt"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")
    if request.get("prompt") != prompt:
        raise ValueError(f"{demo}: request.json and prompt.txt disagree")
    videos = request.get("video_path")
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError(f"{demo}: expected exactly one video")
    video_path = (REPO_ROOT / videos[0]).resolve()
    if not video_path.is_file() or not video_path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{demo}: video must be an existing repository file")
    payload = {"video_path": [str(video_path)], "prompt": prompt, "return_details": True}
    return payload, [request_path, prompt_path, video_path]


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def run_case(
    demo: str, url: str, output_root: Path, timeout: float,
    configuration: dict, source_hashes: dict[str, str],
) -> dict:
    payload, inputs = load_case(demo)
    input_hashes = hash_files(inputs)
    directory = output_root / demo
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "demo": demo,
        "started_at": utc_now(),
        "configuration": configuration,
        "input_sha256": input_hashes,
        "agent_source_sha256": source_hashes,
        "return_details": True,
        "attempts": 1,
    }
    # Retain start-time provenance even if this process is interrupted mid-stream.
    write_json(directory / "started.json", metadata)
    response_record = None
    execution_error = None
    heartbeat_count = 0
    started = time.monotonic()
    try:
        with make_session(url) as session:
            with session.post(url, json=payload, stream=True, timeout=(30, timeout)) as response:
                metadata["http_status"] = response.status_code
                with (directory / "stream.jsonl").open("x", encoding="utf-8") as stream:
                    if not response.ok:
                        body = response.text
                        stream.write(body)
                        try:
                            response_record = json.loads(body)
                        except json.JSONDecodeError:
                            response_record = {"http_status": response.status_code, "body": body}
                        raise RuntimeError(f"HTTP {response.status_code}")
                    for line in response.iter_lines(chunk_size=1):
                        if time.monotonic() - started > timeout:
                            raise TimeoutError(f"evaluation exceeded {timeout:g} seconds")
                        if not line:
                            continue
                        decoded = line.decode("utf-8")
                        stream.write(decoded + "\n")
                        stream.flush()
                        record = json.loads(decoded)
                        if not isinstance(record, dict):
                            raise ValueError("API stream record must be a JSON object")
                        if record.get("status") == "processing" and "index" not in record:
                            heartbeat_count += 1
                            continue
                        if response_record is not None:
                            raise ValueError("API returned more than one final record")
                        response_record = record
                    if response_record is None:
                        raise ValueError("API stream ended without a final result")
        status = response_record.get("status")
        if status not in {"success", "needs_review"}:
            raise RuntimeError(f"API evaluation status: {status!r}")
        if response_record.get("index") != 0:
            raise ValueError("API final result has an unexpected video index")
        if not isinstance(response_record.get("details"), dict):
            raise ValueError("API final result omitted requested details")
        if status == "needs_review" and response_record.get("score") is not None:
            raise ValueError("needs_review response must preserve score: null")
    except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        metadata["completed_at"] = utc_now()
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
        metadata["heartbeat_count"] = heartbeat_count
        try:
            metadata["inputs_unchanged"] = hash_files(inputs) == input_hashes
            metadata["agent_sources_unchanged"] = hash_files(
                sorted((REPO_ROOT / "reward_as_agent").rglob("*.py"))
            ) == source_hashes
            if not metadata["inputs_unchanged"] or not metadata["agent_sources_unchanged"]:
                execution_error = "ProvenanceError: input or Agent source changed during evaluation"
        except OSError as exc:
            execution_error = f"ProvenanceError: {exc}"
        # A missing response stays null; errors are metadata, never invented API scores.
        write_json(directory / "response.json", response_record)
        metadata["status"] = response_record.get("status") if isinstance(response_record, dict) else None
        metadata["execution_error"] = execution_error
        write_json(directory / "metadata.json", metadata)
    summary = {
        "demo": demo,
        "status": metadata["status"],
        "score": response_record.get("score") if isinstance(response_record, dict) else None,
        "execution_error": execution_error,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    available_demos = discover_demos()
    parser.add_argument("--url", default="http://127.0.0.1:7024")
    parser.add_argument("--demo", action="append", choices=available_demos, help="Repeat to select demos; default: all")
    parser.add_argument("--output", required=True, type=Path, help="New results directory; cases are never overwritten")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout", type=float, default=900, help="Per-demo timeout in seconds")
    args = parser.parse_args(argv)
    if not available_demos:
        parser.error("no bundled demos with request.json and prompt.txt were found")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a positive finite number")
    demos = list(dict.fromkeys(args.demo or available_demos))
    try:
        health_url, evaluation_url = service_urls(args.url)
        for demo in demos:
            load_case(demo)
            if (args.output / demo).exists():
                raise ValueError(f"refusing to overwrite {args.output / demo}")
        source_hashes = hash_files(sorted((REPO_ROOT / "reward_as_agent").rglob("*.py")))
        with make_session(health_url) as session:
            response = session.get(health_url, timeout=30)
            response.raise_for_status()
            health = response.json()
        if health.get("status") != "ok" or health.get("pipeline") != "evidence" or health.get("provider") != "doubao":
            raise ValueError("service must be healthy with pipeline=evidence and provider=doubao")
        configuration = {key: health[key] for key in PUBLIC_HEALTH_FIELDS if key in health}
        args.output.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(
                run_case, demo, evaluation_url, args.output, args.timeout, configuration, source_hashes,
            ) for demo in demos]
            results = [future.result() for future in futures]
        errors = sum(result["execution_error"] is not None for result in results)
        print(f"Completed {len(results)} demo(s); execution errors: {errors}; results: {args.output}")
        return 1 if errors else 0
    except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
        print(f"Demo run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
