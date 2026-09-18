"""Check a frozen 13-case diagnostic protocol without inventing human labels."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROTOCOL = "v3-diagnostics-13-v1"
INPUT_SHA256 = "33938aedf5d23afaa58870eebe0277682802c4a924175d7e2d2ee5a112811d0b"
REAL = {
    "Y-08-23_19:49:03_87751d9c/video_0", "Y-08-20_14:13:11_5bf6d507/video_1",
    "Y-08-23_08:42:57_1f54c639/video_1", "Y-08-23_10:35:10_b2e1b146/video_0",
    "Y-08-19_23:53:21_f29290ac/video_1", "Y-08-22_02:17:17_d550b564/video_0",
    "Y-08-20_10:50:09_ef8697db/video_1", "Y-08-20_10:50:09_ef8697db/video_0",
}
TASK_NEGATIVES = {"control/case_01_frozen_first", "control/case_04_frozen_first", "control/case_04_wrong_target"}
JUMP, BLACK = "control/case_04_repeated_state_jumps", "control/case_04_all_black"
EXPECTED = REAL | TASK_NEGATIVES | {JUMP, BLACK}


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("samples", value.get("rows", [value]))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("Expected sample objects in JSON list/JSONL")
    return value


def verify_manifest(rows):
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 13 or set(ids) != EXPECTED:
        raise ValueError("Protocol requires the exact 13 unique development cases, without the repeated visibility source reference")
    inputs = [{key: row[key] for key in ("sample_id", "video_path", "prompt")}
              for row in sorted(rows, key=lambda row: row["sample_id"])]
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != INPUT_SHA256:
        raise ValueError("Frozen sample/path/task identity changed; do not silently replace diagnostic inputs")
    return {row["sample_id"]: row for row in rows}


def number(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1


def verified_failure_zero(row, spec):
    """Recompute a new-policy failure reward before waiving physics-only review.

    A zero score or an asserted training-eligible flag is not proof. Reuse the
    contract checker to validate the frozen task, supplied frame references,
    report-bound audits, any conditional resolution, and the current reducer.
    Old records without the diagnostic/training split keep the frozen behavior.
    This proves logged structural consistency, not visual ground truth.
    """
    details = row.get("details") or {}
    if (not number(row.get("new_score")) or row["new_score"] != 0
            or "diagnostic_score" not in details
            or (details.get("evidence_report") or {}).get("task_assessment", {}).get("verdict") != "failed"):
        return False
    try:
        from scripts.evidence.check_contract_results import check_record

        checked = check_record(
            row, spec, details["task_contract"], details["task_contract_registry_sha256"],
            details["task_contract_review"],
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False
    return ((details.get("scoring") or {}).get("decisive_failure") is True
            and checked["expected_score"] == 0
            and checked["expected_training_eligible"] is True
            and checked["expected_review_required"] is False)


def inspect_record(row, spec):
    sid = spec["sample_id"]
    row = row or {}
    details = row.get("details") or {}
    report = details.get("evidence_report") or {}
    task, physics = report.get("task_assessment") or {}, report.get("physics_assessment") or {}
    status = row.get("status", "missing")
    review = status == "needs_review" or details.get("review_required") is True or (details.get("scoring") or {}).get("review_required") is True
    score = row.get("new_score")
    score = score if number(score) else None
    eligible = status == "success" and row.get("training_eligible") is True and not review and score is not None
    flags = []
    if row.get("new_score") is not None and not number(row["new_score"]):
        flags.append("invalid_numeric_score")
    if row and any(row.get(key) != spec[key] for key in ("video_path", "prompt")):
        flags.append("result_input_differs_from_frozen_manifest")
    if status == "success" and not eligible:
        flags.append("success_but_training_contract_inconsistent")
    if row.get("training_eligible") is True and not eligible:
        flags.append("training_eligible_conflicts_with_status_review_or_score")
    if task.get("verdict") == "complete" and task.get("target_match") != "match":
        flags.append("complete_conflicts_with_target_identity")
    if task.get("verdict") == "mostly_complete" and task.get("target_match") == "mismatch":
        flags.append("mostly_complete_conflicts_with_target_mismatch")
    supplied = {item["source_frame_index"] for item in details.get("frame_manifest", [])
                if isinstance(item, dict) and type(item.get("source_frame_index")) is int}
    observations = {item.get("id"): item for item in report.get("observations", []) if isinstance(item, dict)}
    independent_failure_zero = eligible and verified_failure_zero(row, spec)
    issue_audit, motion = [], []
    for issue in physics.get("issues", []):
        refs = issue.get("evidence", [])
        frames = {i for ref in refs for i in observations.get(ref, {}).get("frames", []) if type(i) is int}
        unknown = [ref for ref in refs if ref not in observations]
        if unknown or (supplied and not frames <= supplied):
            flags.append("physics_issue_has_unknown_observation_or_unsupplied_frame")
        if issue.get("certainty") == "confirmed" and len(frames) < 2:
            flags.append("confirmed_issue_lacks_two_distinct_referenced_frames")
        if issue.get("certainty") == "uncertain" and eligible and not independent_failure_zero:
            flags.append("uncertain_physics_issue_not_routed_to_review")
        supported_reference = not unknown and len(frames) >= 2 and bool(supplied) and frames <= supplied
        if issue.get("kind") == "motion" and issue.get("certainty") == "confirmed" and supported_reference:
            motion.append(issue)
        issue_audit.append({"kind": issue.get("kind"), "certainty": issue.get("certainty"),
                            "severity": issue.get("severity"), "observation_refs": refs,
                            "reason": issue.get("reason"), "alternative_explanation": issue.get("alternative_explanation"),
                            "referenced_frame_union": sorted(frames),
                            "referenced_observations": [observations.get(ref) for ref in refs],
                            "issue_specific_multiframe_support_verified": False,
                            "semantic_review_required": True,
                            "limit": "A multi-frame observation bundle does not prove this particular defect in multiple frames."})
    confirmed = [issue for issue in physics.get("issues", []) if issue.get("certainty") == "confirmed"]
    if physics.get("verdict") in ("plausible", "unobservable") and confirmed:
        flags.append("physics_verdict_conflicts_with_confirmed_issue")
    draft = {item.get("id"): item for item in (details.get("draft_report") or {}).get("observations", []) if isinstance(item, dict)}
    changed = [eid for eid in observations if eid in draft and observations[eid] != draft[eid]]
    trace = details.get("trace") or []
    protection = review and row.get("training_eligible") is False
    check, reason = None, "Real-video truth is unknown; no correctness check is defined."
    if sid not in REAL:
        if status in ("error", "missing") or not report:
            reason = "Unresolved: missing/failed evaluation is not a passing control."
        elif sid in TASK_NEGATIVES:
            check = task.get("verdict") == "failed"
            reason = "Predefined task-category check only; no exact numeric reward is prescribed."
        elif sid == BLACK:
            check = row.get("training_eligible") is False and (task.get("verdict") == "unobservable" or protection)
            reason = "Black input must be unobservable or explicitly reviewed, and must not be training-eligible."
        else:
            detected = bool(motion)
            unconditional_full = eligible and score == 1 and not detected
            check = (detected or protection) and not unconditional_full
            reason = "Require a referenced structured motion anomaly or explicit nontraining review; no unique physics severity/category is prescribed."
        if flags:
            check = False
            reason += " Structural/transport inconsistencies invalidate an otherwise passing check."
    total = (details.get("video_metadata") or {}).get("decoded_frames")
    return {
        "sample_id": sid, "kind": "real_unknown_truth" if sid in REAL else "constructed_behavior_check",
        "status": status, "score": score, "training_eligible": eligible, "review_required": review,
        "task_verdict": task.get("verdict"), "target_match": task.get("target_match"), "physics_verdict": physics.get("verdict"),
        "behavior_check_pass": check, "behavior_check_reason": reason, "structural_flags": sorted(set(flags)),
        "issue_evidence_audit": issue_audit, "draft_to_final_changed_observation_ids": changed,
        "real_accuracy": None, "manual_wording_and_visible_evidence_review_required": sid in REAL or bool(issue_audit),
        "verification_frames": {"supplied_count": len(supplied), "decoded_count": total,
                                 "all_original_frames_supplied": supplied == set(range(total)) if type(total) is int and total > 0 else None},
        "stage_request_provenance": [{key: entry.get(key) for key in ("stage", "attempt", "messages_sha256", "response_id", "cache_reused", "validation_error", "error")} for entry in trace],
        "schema_validation_errors": sum(bool(entry.get("validation_error")) for entry in trace),
        "evaluator_version": row.get("evaluator_version"), "run_id": row.get("run_id"),
        "source_video_sha256": row.get("source_video_sha256"),
        "explicitly_uncached": row.get("cache_reused") is False and not any(entry.get("cache_reused") is True for entry in trace),
    }


def check_runs(manifest_rows, batches):
    manifest = verify_manifest(manifest_rows)
    grouped, outside = defaultdict(list), []
    for source, rows in batches:
        for index, row in enumerate(rows):
            if row.get("sample_id") not in manifest:
                outside.append({"source": source, "record_index": index, "sample_id": row.get("sample_id")})
                continue
            group = row.get("run_id") or "missing-run-id:" + source
            grouped[group].append((source, index, row))
        if not rows:
            grouped["empty-results:" + source] = []
    runs = []
    for group, records in grouped.items():
        latest = {row["sample_id"]: row for _, _, row in records}
        cases = [inspect_record(latest.get(sid), spec) for sid, spec in manifest.items()]
        controls = [case for case in cases if case["kind"] == "constructed_behavior_check"]
        subsets = {}
        for name, selected in (("real", [c for c in cases if c["sample_id"] in REAL]), ("controls", controls)):
            subsets[name] = {"expected": len(selected), "status_counts": dict(Counter(c["status"] for c in selected)),
                             "training_eligible_count": sum(c["training_eligible"] for c in selected),
                             "training_eligible_coverage": sum(c["training_eligible"] for c in selected) / len(selected)}
        runs.append({"group": group, "run_ids": sorted({c["run_id"] for c in cases if c["run_id"]}),
                     "evaluator_versions": sorted({c["evaluator_version"] for c in cases if c["evaluator_version"]}),
                     "coverage": subsets,
                     "constructed_checks": {"expected": 5, "pass": sum(c["behavior_check_pass"] is True for c in controls),
                                            "fail": sum(c["behavior_check_pass"] is False for c in controls),
                                            "unresolved": sum(c["behavior_check_pass"] is None for c in controls)},
                     "cases": cases,
                     "all_attempts": [{"source": source, "record_index": index, "sample_id": row["sample_id"], "status": row.get("status"), "score": row.get("new_score") if number(row.get("new_score")) else None, "schema_validation_errors": sum(bool(t.get("validation_error")) for t in (row.get("details") or {}).get("trace", []))} for source, index, row in records]})
    repeats = []
    for a, b in itertools.combinations(runs, 2):
        provenance = (len(a["evaluator_versions"]) == 1 and a["evaluator_versions"] == b["evaluator_versions"]
                      and len(a["run_ids"]) == len(b["run_ids"]) == 1 and a["run_ids"] != b["run_ids"])
        paired = [(x, y) for x, y in zip(a["cases"], b["cases"]) if provenance and x["explicitly_uncached"] and y["explicitly_uncached"]
                  and isinstance(x["source_video_sha256"], str) and len(x["source_video_sha256"]) == 64
                  and all(c in '0123456789abcdefABCDEF' for c in x["source_video_sha256"])
                  and x["source_video_sha256"] == y["source_video_sha256"]
                  and x["status"] not in ("missing", "error") and y["status"] not in ("missing", "error")
                  and not x["structural_flags"] and not y["structural_flags"]]
        subsets = {}
        for name, ids in (("real", REAL), ("controls", EXPECTED - REAL)):
            all_pairs = [(x, y) for x, y in paired if x["sample_id"] in ids]
            valid = [(x, y) for x, y in all_pairs if x["training_eligible"] and y["training_eligible"]]
            diffs = [abs(x["score"] - y["score"]) for x, y in valid]
            tasks = [(x, y) for x, y in valid if x["task_verdict"] is not None and y["task_verdict"] is not None]
            subsets[name] = {"independent_provenance_pairs": len(all_pairs), "both_training_eligible": len(valid),
                             "mean_absolute_score_difference": statistics.mean(diffs) if diffs else None,
                             "max_absolute_score_difference": max(diffs, default=None),
                             "difference_gt_0_1_count": sum(d > .1 + 1e-9 for d in diffs),
                             "task_verdict_agreement": {"same": sum(x["task_verdict"] == y["task_verdict"] for x, y in tasks), "denominator": len(tasks)}}
        repeats.append({"first_group": a["group"], "second_group": b["group"],
                        "same_version_distinct_run_ids": provenance, "subsets": subsets})
    return {"protocol": PROTOCOL, "frozen_input_sha256": INPUT_SHA256, "expected_unique_cases": 13,
            "outside_manifest_records": outside, "runs": runs, "repeatability": repeats,
            "human_accuracy_established": False, "real_video_accuracy": None,
            "limits": ["Constructed behavior checks are not human labels or natural-video accuracy.",
                       "Multi-frame reference counts and unchanged observation IDs do not prove semantic grounding; wording must be checked against actual frames.",
                       "Fresh repeated-input comparisons additionally require matching nonempty source_video_sha256; matching paths alone do not prove equal source bytes. messages_sha256 hashes messages, not all HTTP request parameters.",
                       "All provided runs and attempt outcomes are retained; do not select only improved samples. Review remains distinct from a usable reward."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON file")
    args = parser.parse_args()
    report = check_runs(read_rows(args.manifest), [(str(path), read_rows(path)) for path in args.results])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": PROTOCOL, "runs": len(report["runs"]), "human_accuracy_established": False}))


if __name__ == "__main__":
    main()
