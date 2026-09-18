#!/usr/bin/env python3
"""Replay only the current reward reducer over retained, validated demo evidence.

This performs no model or tool calls. It cannot create a missing visual failure
resolution and never claims that the retained evaluation used the new Agent.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reward_as_agent.evidence_grounding import (  # noqa: E402
    core_report, observation_time_manifest, validate_grounded_report,
)
from reward_as_agent.evidence_schema import score_report  # noqa: E402
from reward_as_agent.requirement_audit import validate_requirement_audit  # noqa: E402
from reward_as_agent.task_contract import validate_requirement_checks, validate_task_contract  # noqa: E402
from reward_as_agent.training_reward import (  # noqa: E402
    SCORING_VERSION, apply_training_reward, needs_failure_resolution, validate_failure_resolution,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same(actual, expected, name):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{name}: retained artifact is inconsistent")


def validate_source(response: dict, run: dict) -> None:
    """Verify frozen contract, frame references, both scope audits and old reducer."""
    if response.get("status") not in {"success", "needs_review"}:
        raise ValueError("Only completed evaluations can be replayed; execution errors remain errors")
    details = response["details"]
    if any("policy_replay" in item for item in (response, details, run)) or "original_run" in run:
        raise ValueError("Repeat policy replay is not supported")
    if "diagnostic_score" in details or details["scoring"].get("scoring_version") == SCORING_VERSION:
        raise ValueError("Source already uses the current training reward policy")
    if details.get("error") and not details["error"].startswith("needs_review:"):
        raise ValueError("Execution errors cannot be replayed as rewards")
    if not details.get("evaluator_version"):
        raise ValueError("Missing original evaluator_version")
    if run.get("evaluator_version"):
        same(run["evaluator_version"], details["evaluator_version"], "evaluator_version")
    contract, report = details["task_contract"], details["evidence_report"]
    validate_task_contract(contract)
    same(details["task_contract_sha256"], contract["contract_sha256"], "contract hash")
    frames, metadata = details["frame_manifest"], details["video_metadata"]
    fps, count = metadata["fps"], metadata["decoded_frames"]
    if (type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0
            or type(count) is not int or count < 1 or not isinstance(frames, list) or not frames):
        raise ValueError("Invalid decoded video metadata or frame manifest")
    ids = []
    for frame in frames:
        index = frame["source_frame_index"]
        if type(index) is not int or not 0 <= index < count or index in ids:
            raise ValueError("Invalid or duplicate source frame index")
        same(frame["timestamp_seconds"], round(index / fps, 4), "frame timestamp")
        ids.append(index)
    validate_grounded_report(report, ids, contract)
    summary = validate_requirement_checks(report["requirement_checks"], contract, report)
    same(details["requirement_summary"], summary, "requirement summary")
    same(details["observation_time_manifest"], observation_time_manifest(report, frames), "observation times")
    audits = details["requirement_scope_audits"]
    if not isinstance(audits, list) or len(audits) not in (1, 2):
        raise ValueError("Require one scope audit, with at most one repair and re-audit")
    initial = details["pre_scope_repair_report"]
    validate_grounded_report(initial, ids, contract)
    for index, entry in enumerate(audits):
        audited = initial if index == 0 else report
        same(entry["report_sha256"], digest(json.dumps(
            audited, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()), "scope audited report hash")
        validate_requirement_audit(entry["audit"], contract, audited)
    issues = any(check["issues"] for check in audits[0]["audit"]["checks"])
    if (len(audits) == 2) != issues:
        raise ValueError("Scope issues require a retained repair and re-audit")
    if len(audits) == 1:
        same(initial, report, "final audited report")
    resolution = details.get("failure_reward_resolution")
    if resolution is not None:
        validate_failure_resolution(resolution, report, contract, audits, ids)
        accepted = [entry for entry in details["trace"]
                    if entry.get("stage") == "failure_reward_resolution"
                    and not entry.get("validation_error") and not entry.get("error")]
        if not accepted:
            raise ValueError("Failure resolution requires the original image-grounded trace")
        same(accepted[-1]["output"], resolution, "failure resolution trace")
        for field in ("response_id", "messages_sha256", "request_payload_sha256"):
            if not accepted[-1].get(field):
                raise ValueError("Failure resolution trace lacks request provenance")
    elif needs_failure_resolution(report, contract, audits):
        raise ValueError("Task ambiguity needs a real image-grounded failure resolution; cannot replay offline")

    # Limit this migration to the published default/WMReward reflect demos. Other
    # optional protocol reducers need their own replay, not guessed reconstruction.
    expected = score_report(core_report(report))
    expected["scoring_version"] = "evidence-soft-physics-v2-with-scope-review-v5-provisional"
    physics = details.get("physics_evidence") or {}
    if physics.get("input_unobservable"):
        expected["total_score"] = None
        expected["review_reasons"].append(
            "input visibility: all decoded frames are spatially uniform; no grounded manipulation evidence")
        expected["input_visibility_gate"] = physics["input_visibility"]
    expected["review_reasons"].extend(
        f"task requirement {item}: unresolved visible evidence" for item in summary["unresolved_requirement_ids"])
    if details.get("task_contract_review"):
        expected["review_reasons"].append("task contract: " + details["task_contract_review"]["reason"])
    for check in audits[-1]["audit"]["checks"]:
        for issue in check["issues"]:
            expected["review_reasons"].append(
                f"task requirement {check['requirement_id']}: unresolved scope audit "
                f"({issue['kind']}): {issue['reason']}")
    expected["review_required"] = bool(expected["review_reasons"])
    same(details["scoring"], expected, "old reducer (only gate_policy=off supported)")
    old_score = expected["total_score"]
    same(details["total_score"], old_score if old_score is not None else -1, "old transport score")
    same(details["review_required"], expected["review_required"], "old review gate")
    same(details["training_eligible"], old_score is not None and not expected["review_required"], "old eligibility")
    same(response["status"], "needs_review" if expected["review_required"] else "success", "old status")
    same(response["score"], None if expected["review_required"] else old_score, "old API score")


def replay(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output == source or output.is_relative_to(source):
        raise ValueError("Output must be a new directory outside the source")
    names = ["response.json", "run.json"] + [name for name in ("tools.json", "reports.json")
                                             if (source / name).is_file()]
    originals = {name: (source / name).read_bytes() for name in names}
    response, run = (json.loads(originals[name]) for name in ("response.json", "run.json"))
    validate_source(response, run)
    if "tools.json" in originals:
        same(json.loads(originals["tools.json"]), response["details"].get("physics_evidence"), "tool evidence snapshot")
    if "reports.json" in originals:
        reports = json.loads(originals["reports.json"])
        same(reports["final_after_scope_audit"], response["details"]["evidence_report"], "final report snapshot")
        same(reports["after_tool_reflection"], response["details"]["pre_scope_repair_report"], "reflected report snapshot")
        same(reports["before_tool_reflection"],
             (response["details"].get("physics_evidence") or {}).get("pre_tool_report"), "pre-tool report snapshot")
    # Preserve the original evaluator's input binding, where recorded, and reject
    # a changed local video/prompt rather than attach old evidence to new inputs.
    for name, expected_hash in run.get("input_sha256", {}).items():
        path = (REPO_ROOT / name).resolve()
        if not path.is_relative_to(REPO_ROOT) or digest(path.read_bytes()) != expected_hash:
            raise ValueError(f"Original evaluation input hash mismatch: {name}")
    source_files = [Path(__file__).resolve()] + sorted((REPO_ROOT / "reward_as_agent").glob("*.py"))
    policy_hashes = {str(path.relative_to(REPO_ROOT)): digest(path.read_bytes()) for path in source_files}
    details = copy.deepcopy(response["details"])
    scoring = apply_training_reward(
        details["scoring"], details["evidence_report"], details["task_contract"],
        details["requirement_scope_audits"], contract_review=details.get("task_contract_review"),
        physics_evidence=details.get("physics_evidence"),
        failure_resolution=details.get("failure_reward_resolution"),
    )
    score = scoring["total_score"]
    details.update(scoring=scoring, total_score=score if score is not None else -1,
                   review_required=scoring["review_required"],
                   training_eligible=score is not None and not scoring["review_required"],
                   failure_reward_resolution=details.get("failure_reward_resolution"))
    for field in ("diagnostic_score", "diagnostic_review_required", "diagnostic_review_reasons"):
        details[field] = scoring[field]
    details.pop("error", None)
    if score is None:
        details["error"] = "needs_review: insufficient observable evidence for numerical reward"
    policy = {"schema_version": "demo-reward-policy-replay-v1",
              "execution_mode": "offline_reward_policy_replay",
              "replayed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scoring_version": SCORING_VERSION,
              "original_evaluator_version": details["evaluator_version"],
              "policy_source_sha256": policy_hashes,
              "source_artifact_sha256": {name: digest(data) for name, data in originals.items()},
              "original_status": response["status"], "original_score": response["score"],
              "original_provisional_score": response.get("provisional_score"),
              "new_model_calls": 0, "new_tool_calls": 0,
              "retained_evidence_unchanged": True,
              "note": "Original inference and tool records retained; only the deterministic reward policy was reapplied."}
    details["policy_replay"] = copy.deepcopy(policy)
    from reward_as_agent.app import response_from_result
    updated = response_from_result(details)
    updated["details"] = details
    for field in ("model", "provider"):
        if field in response:
            updated[field] = response[field]
    new_run = {"execution_mode": "offline_reward_policy_replay", "status": updated["status"],
               "score": updated["score"], "evaluator_version": details["evaluator_version"],
               "policy_replay": policy, "original_run": run}
    if originals != {name: (source / name).read_bytes() for name in names}:
        raise ValueError("Source artifacts changed during replay")
    if policy_hashes != {str(path.relative_to(REPO_ROOT)): digest(path.read_bytes()) for path in source_files}:
        raise ValueError("Policy source changed during replay")
    # Exclusive creation ensures validation failures never alter inputs/outputs.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in originals.items():
        destination = "source_" + name if name in ("response.json", "run.json") else name
        with (output / destination).open("xb") as stream:
            stream.write(data)
    for name, value in (("response.json", updated), ("run.json", new_run)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {"output": str(output), "status": updated["status"], "score": updated["score"],
            "original_score": response["score"], "new_model_calls": 0, "new_tool_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(replay(args.source, args.output), ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
