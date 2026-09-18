#!/usr/bin/env python3
"""Run a bundled demo with the existing live WMReward reflection integration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = REPO_ROOT / "scripts" / "frozen_v41"
for import_root in (REPO_ROOT, WORKER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_demos import DEMOS, hash_files, load_case, sha256_file, utc_now, write_json


def implementation_paths() -> list[Path]:
    return sorted((REPO_ROOT / "reward_as_agent").rglob("*.py")) + sorted(WORKER_ROOT.rglob("*.py"))


def worker_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.worker_python.absolute()), "-m", "scripts.frozen_v41.wmreward_worker",
        "--repo", str(args.wmreward_repo.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--device", args.device,
    ]


def inspect_tool_completion(details: dict, trace: list, source_sha256: str) -> tuple[dict | None, list[str]]:
    evidence = details.get("physics_evidence")
    if not isinstance(evidence, dict):
        evidence = next((entry for entry in reversed(trace)
                         if entry.get("stage") == "independent_physics_tools"), None)
    errors = []
    if not evidence:
        return None, ["WMReward integration did not finish"]
    records = evidence.get("tool_records", [])
    if len(records) != 1 or records[0].get("worker") != "wmreward":
        errors.append("expected exactly one live WMReward result")
    for record in records:
        result = record.get("result", {})
        score = result.get("raw_score")
        if result.get("status") != "ok":
            errors.append("WMReward status is not ok")
        if type(score) not in (int, float) or not math.isfinite(score):
            errors.append("WMReward omitted a finite raw score")
        if result.get("video_sha256") != source_sha256:
            errors.append("WMReward source hash does not match the video")
        if result.get("fallback_to_baseline") or result.get("cache_hit"):
            errors.append("WMReward used a fallback or a cached result")
    if evidence.get("mode") != "reflect" or evidence.get("fallback_to_baseline"):
        errors.append("tool integration fell back instead of completing reflection")
    reflected = any(entry.get("stage") == "physics_tool_reflection"
                    and isinstance(entry.get("output"), dict)
                    and not entry.get("validation_error") and not entry.get("error")
                    for entry in trace)
    if evidence.get("reflection_applied") is not True or not reflected:
        errors.append("image-grounded tool reflection did not execute successfully")
    return evidence, errors


async def run(args: argparse.Namespace, demo: str) -> int:
    from reward_as_agent.config import get_settings
    from reward_as_agent.evidence_pipeline import EvidencePipeline
    from reward_as_agent.pipeline import process_one_video_safe
    from scripts.frozen_v41.physics_integration import PhysicsEvidenceHook
    from scripts.frozen_v41.worker_client import WorkerClient

    payload, inputs = load_case(demo)
    input_hashes = hash_files(inputs)
    source_hashes = hash_files(implementation_paths())
    settings = get_settings()
    if settings.provider != "doubao":
        raise ValueError("this demonstration requires the current Doubao Agent")
    if os.environ.get("REWARD_GATE_POLICY", "off") != "off":
        raise ValueError("leave REWARD_GATE_POLICY=off for this tool-only demonstration")
    if not args.worker_python.is_file() or not os.access(args.worker_python, os.X_OK):
        raise ValueError("--worker-python must identify an executable Python interpreter")
    if not args.wmreward_repo.is_dir() or not args.checkpoint.is_file():
        raise ValueError("WMReward repository and checkpoint must already exist locally")
    actual_checkpoint_sha256 = await asyncio.to_thread(sha256_file, args.checkpoint)
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 does not match --checkpoint-sha256")

    output = (args.output / demo).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = worker_command(args)
    previous_env = {key: os.environ.get(key) for key in (
        "PYTHONPATH", "REWARD_EVIDENCE_TRACE_DIR", "REWARD_AS_AGENT_CACHE_BYPASS",
    )}
    import_paths = [str(REPO_ROOT), str(WORKER_ROOT)]
    if previous_env["PYTHONPATH"]:
        import_paths.append(previous_env["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(import_paths)
    os.environ["REWARD_EVIDENCE_TRACE_DIR"] = str(output / "traces")
    os.environ["REWARD_AS_AGENT_CACHE_BYPASS"] = "1"
    metadata = {
        "demo": demo,
        "started_at": utc_now(),
        "execution_mode": "live_local_wmreward_with_doubao_reflection",
        "configuration": {key: getattr(settings, key) for key in (
            "provider", "model", "temperature", "max_tokens", "llm_timeout", "max_retries",
            "dp_size", "max_inflight_per_dp",
        )},
        "input_sha256": input_hashes,
        "agent_and_tool_source_sha256": source_hashes,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "worker_command": command,
        "worker_timeout_seconds": 900,
        "mode": "reflect",
        "gate_policy": "off",
        "application_cache_bypassed": True,
        "attempts": 1,
    }
    write_json(output / "started.json", metadata)
    client = WorkerClient(command, output / "wmreward.stderr.log", timeout_s=900, startup_timeout_s=900)
    response = None
    details = {}
    evidence = None
    trace = []
    errors = []
    started = time.monotonic()
    try:
        hook = PhysicsEvidenceHook({"wmreward": client}, mode="reflect")
        pipeline = EvidencePipeline(settings, physics_hook=hook)
        metadata.update(
            evaluator_version=pipeline.evaluator_version,
            gate_policy=pipeline.gate_policy,
            audit_mode=pipeline.audit_mode,
            frame_budget=pipeline.frame_budget,
            physics_completion_required=pipeline.physics_completion_required,
            worker_requests=hook.request_specs,
        )
        metadata["configuration"]["effective_max_tokens"] = pipeline.settings.max_tokens
        # Reuse the service's exact score/null and provisional-score conversion.
        from reward_as_agent.app import response_from_result
        details = await process_one_video_safe(pipeline, payload["video_path"][0], payload["prompt"], 0)
        response = response_from_result(details)
        response.update(details=details, model=settings.model, provider=settings.provider)
        if response["status"] not in {"success", "needs_review"}:
            errors.append("Agent evaluation did not return a completed result")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            await client.close()
        except Exception as exc:
            errors.append(f"worker cleanup failed: {type(exc).__name__}: {exc}")
        try:
            trace = details.get("trace", [])
            if not trace:
                identity = hashlib.sha256((payload["video_path"][0] + "\n" + payload["prompt"]).encode()).hexdigest()[:20]
                trace_path = output / "traces" / (identity + ".json")
                if trace_path.is_file():
                    trace = json.loads(trace_path.read_text(encoding="utf-8")).get("trace", [])
            source_sha256 = input_hashes[str(inputs[-1].relative_to(REPO_ROOT))]
            evidence, tool_errors = inspect_tool_completion(details, trace, source_sha256)
            errors.extend(tool_errors)
            metadata["inputs_unchanged"] = hash_files(inputs) == input_hashes
            metadata["agent_and_tool_sources_unchanged"] = hash_files(implementation_paths()) == source_hashes
            metadata["checkpoint_unchanged"] = await asyncio.to_thread(sha256_file, args.checkpoint) == actual_checkpoint_sha256
            for field in ("inputs_unchanged", "agent_and_tool_sources_unchanged", "checkpoint_unchanged"):
                if not metadata[field]:
                    errors.append(f"provenance check failed: {field}")
            records = (evidence or {}).get("tool_records", [])
            for record in records:
                if actual_checkpoint_sha256 not in record.get("result", {}).get("model_revision", ""):
                    errors.append("WMReward model revision does not identify the requested checkpoint")
        except Exception as exc:
            errors.append(f"artifact verification failed: {type(exc).__name__}: {exc}")
        finally:
            for key, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        reflected_report = next((entry["output"] for entry in reversed(trace)
                                 if entry.get("stage") == "physics_tool_reflection"
                                 and isinstance(entry.get("output"), dict)
                                 and not entry.get("validation_error") and not entry.get("error")), None)
        metadata.update(
            completed_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            reflection_applied=(evidence or {}).get("reflection_applied") is True,
            fallback_to_baseline=(evidence or {}).get("fallback_to_baseline"),
            status=(response or {}).get("status"),
            execution_errors=errors,
            tool_demo_completed=not errors,
        )
        write_json(output / "response.json", response)
        write_json(output / "tools.json", evidence)
        write_json(output / "trace.json", trace)
        write_json(output / "reports.json", {
            "before_tool_reflection": (evidence or {}).get("pre_tool_report"),
            "after_tool_reflection": reflected_report,
            "final_after_scope_audit": details.get("evidence_report"),
        })
        write_json(output / "metadata.json", metadata)
    print(json.dumps({
        "demo": demo, "status": metadata["status"],
        "score": (response or {}).get("score"),
        "reflection_applied": metadata["reflection_applied"],
        "tool_demo_completed": metadata["tool_demo_completed"], "errors": errors,
        "output": str(output),
    }, ensure_ascii=False), flush=True)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="append", choices=DEMOS,
                        help="Select demos; repeat this option. Defaults to all demos.")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is never overwritten")
    parser.add_argument("--worker-python", required=True, type=Path)
    parser.add_argument("--wmreward-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True, type=str.lower)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
        parser.error("--checkpoint-sha256 must be a 64-character SHA256 digest")
    if args.output.exists():
        parser.error("--output must not already exist")
    try:
        demos = list(dict.fromkeys(args.demo or DEMOS))
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        for demo in demos:
            result = asyncio.run(run(args, demo))
            results.append(result)
        return max(results, default=0)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"WMReward demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
