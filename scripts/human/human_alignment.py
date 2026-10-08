"""Freeze grouped evaluation sets and measure against explicitly sourced human labels.

Only the Python standard library is required. This module never creates annotations
and never treats agreement between two AI judges as human agreement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


RUBRIC_VERSION = "embodied-human-v1"
RUBRIC = {
    "version": RUBRIC_VERSION,
    "overall": "根据任务要求判断视频中实际可见的对象、动作、最终状态及物理执行是否可信。独立给出整体分数，不要将子项分数相乘。",
    "anchors": [
        {"score": 0, "text": "不可用或明显无关；没有有效任务进展，或物理不可能的过程使整个动作失效。"},
        {"score": 0.25, "text": "只有少量进展；主要任务失败，或明显的严重错误占据主导。"},
        {"score": 0.5, "text": "已有较多进展，但重要要求没有满足，或清楚可见的物理缺陷显著影响结果。"},
        {"score": 0.75, "text": "主要任务成功，仍有明确但影响有限的遗漏或缺陷。"},
        {"score": 1, "text": "可见证据表明任务及要求的最终状态均已满足，没有有意义的可见缺陷。"},
    ],
    "evidence": "完整观看视频，记录可见证据并尽量注明时间。遮挡、正常夹持、运动模糊、透视重叠和柔软物体受力，均不能单独证明穿模或非物理形变。不要臆测不可见的最终状态。物理判断不明确时选择不确定，证据不足时降低置信度。",
    "task_completion": "0=none, 0.25=limited, 0.5=partial, 0.75=mostly, 1=complete; null=not observable.",
    "physics_severity": "none=none observed; minor=local cosmetic defect; moderate=clear defect affecting execution; severe=impossible motion/state invalidates execution; uncertain=evidence insufficient.",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unit_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Score must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Score must be finite and in [0, 1]")
    return score


def latest_successful(rows: list[dict]) -> list[dict]:
    # A later failed retry does not create a valid scored case. We require the
    # latest authoritative record to be successful.
    latest = {row["sample_id"]: row for row in rows}
    result = []
    for sample_id, row in sorted(latest.items()):
        if row.get("status") == "success":
            unit_score(row["new_score"])
            result.append(row)
    return result


def reserved_sample_ids(paths: list[Path]) -> set[str]:
    reserved = set()
    for path in paths:
        value = json.loads(path.read_text())
        items = value if isinstance(value, list) else [value]
        reserved.update(item["sample_id"] for item in items)
    return reserved


def freeze_manifest(rows: list[dict], reserved: set[str], seed: str = "20260910", heldout_fraction: float = .3) -> dict:
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be between 0 and 1")
    selected = latest_successful(rows)
    if not selected:
        raise ValueError("No latest successful results")
    samples = []
    for row in selected:
        sample_id = row["sample_id"]
        pair_id, sep, video_id = sample_id.rpartition("/")
        path = Path(row["video_path"])
        if not sep or path.parent.name != pair_id or path.stem != video_id:
            raise ValueError(f"Inconsistent original pair/video identity: {sample_id}")
        if not path.is_absolute() or path.suffix.lower() != ".mp4":
            raise ValueError(f"Expected absolute MP4 path: {sample_id}")
        samples.append({
            "sample_id": sample_id, "pair_id": pair_id, "video_path": str(path),
            "prompt": row["prompt"],
            "review_id": hashlib.sha256((seed + "\0review\0" + sample_id).encode()).hexdigest()[:24],
        })
    all_ids = {r["sample_id"] for r in samples}
    missing = sorted(reserved - all_ids)
    if missing:
        raise ValueError(f"Reserved inspected cases are missing from the successful set: {missing}")
    groups = sorted({r["pair_id"] for r in samples})
    reserved_groups = {r["pair_id"] for r in samples if r["sample_id"] in reserved}
    candidates = [group for group in groups if group not in reserved_groups]
    candidates.sort(key=lambda group: hashlib.sha256((seed + "\0split\0" + group).encode()).hexdigest())
    n_heldout = min(len(candidates), max(1, round(len(groups) * heldout_fraction)))
    if n_heldout == 0:
        raise ValueError("No uninspected groups remain for held-out evaluation")
    heldout = set(candidates[:n_heldout])
    for row in samples:
        row["split"] = "heldout" if row["pair_id"] in heldout else "development"
        row["previously_inspected"] = row["sample_id"] in reserved
    payload = {
        "schema_version": 1, "rubric": RUBRIC, "seed": seed,
        "requested_heldout_fraction": heldout_fraction,
        "grouping": "original reward_inputs folder: both available candidate videos stay together",
        "reserved_inspected_sample_ids": sorted(reserved),
        "sample_count": len(samples), "pair_count": len(groups),
        "counts_by_split": dict(Counter(r["split"] for r in samples)),
        "samples": samples,
        "scope": "The current successfully evaluated sample, not the full training population; no inferred human labels. Freeze before tuning and keep held-out judgments hidden until final evaluation.",
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def validate_manifest(manifest: dict) -> dict:
    expected = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(canonical(payload)).hexdigest() != expected:
        raise ValueError("Manifest digest mismatch; do not silently change a frozen benchmark")
    if manifest.get("schema_version") != 1 or manifest.get("rubric", {}).get("version") != RUBRIC_VERSION:
        raise ValueError("Unsupported benchmark/rubric version")
    seen, review_ids, splits = set(), set(), {}
    for row in manifest["samples"]:
        if row["sample_id"] in seen or row["review_id"] in review_ids:
            raise ValueError("Duplicate sample/review identifier")
        if row["split"] not in ("development", "heldout"):
            raise ValueError("Invalid split")
        if row["pair_id"] in splits and splits[row["pair_id"]] != row["split"]:
            raise ValueError("Original candidate pair leaks across splits")
        if row["previously_inspected"] and row["split"] != "development":
            raise ValueError("Previously inspected sample cannot be held out")
        seen.add(row["sample_id"])
        review_ids.add(row["review_id"])
        splits[row["pair_id"]] = row["split"]
    return manifest


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[start]] == values[ordered[stop]]:
            stop += 1
        for index in ordered[start:stop]:
            result[index] = (start + 1 + stop) / 2
        start = stop
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denominator if denominator else None


def preference_accuracy(items: list[tuple[str, str, float, float]], within_group: bool, human_tie: float = .05) -> dict:
    correct = total = model_ties = 0
    for left, right in itertools.combinations(items, 2):
        if within_group and left[1] != right[1]:
            continue
        human_delta, model_delta = left[2] - right[2], left[3] - right[3]
        if abs(human_delta) <= human_tie:
            continue
        total += 1
        if abs(model_delta) <= 1e-9:
            model_ties += 1
        elif human_delta * model_delta > 0:
            correct += 1
    return {"non_tied_human_pairs": total, "model_ties": model_ties,
            "strict_accuracy": correct / total if total else None,
            "accuracy_half_credit_model_ties": (correct + .5 * model_ties) / total if total else None,
            "human_tie_threshold": human_tie}


def bootstrap_mae(items: list[tuple[str, str, float, float]], repetitions: int = 1000) -> list[float] | None:
    groups = defaultdict(list)
    for _, group, human, model in items:
        groups[group].append(abs(human - model))
    if len(groups) < 2:
        return None
    values = list(groups.values())
    rng = random.Random(20260910)
    draws = sorted(statistics.mean(error for selected in rng.choices(values, k=len(values)) for error in selected)
                   for _ in range(repetitions))
    return [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]]


def evaluate(manifest: dict, label_rows: list[dict], prediction_rows: list[dict], score_field: str = "new_score", model_run: str | None = None) -> dict:
    validate_manifest(manifest)
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    labels, excluded = {}, Counter()
    # Latest declaration wins, including revoking an earlier independence claim.
    latest_labels = {}
    for row in label_rows:
        key = (row.get("sample_id"), row.get("rater_id"))
        latest_labels[key] = row
    for (sample_id, rater), row in latest_labels.items():
        if sample_id not in samples:
            excluded["unknown_sample"] += 1
        elif not isinstance(rater, str) or not rater.strip():
            excluded["missing_rater"] += 1
        elif row.get("label_source") != "human":
            excluded["non_human_source"] += 1
        elif row.get("independent") is not True or row.get("blinded") is not True:
            excluded["not_independent_and_blinded"] += 1
        elif row.get("manifest_sha256") != manifest["manifest_sha256"] or row.get("rubric_version") != RUBRIC_VERSION:
            excluded["wrong_manifest_or_rubric"] += 1
        else:
            try:
                labels[(sample_id, rater)] = unit_score(row["score"])
            except (ValueError, KeyError):
                excluded["invalid_score"] += 1
    human_scores = defaultdict(list)
    for (sample_id, rater), score in labels.items():
        human_scores[sample_id].append((rater, score))
    # Explicit run IDs make independent repeated calls distinguishable from
    # retried/appended records. Missing IDs form one legacy run, never repeats.
    predictions, provisional_predictions, statuses, failed_predictions = {}, {}, {}, Counter()
    for row in prediction_rows:
        if row.get("sample_id") not in samples:
            continue
        run = str(row.get("run_id", "legacy"))
        key = (row["sample_id"], run)
        predictions.pop(key, None)
        provisional_predictions.pop(key, None)
        statuses[key] = row.get("status", "success")
        if statuses[key] not in ("success", "needs_review"):
            failed_predictions[run] += 1
            continue
        try:
            values = (unit_score(row[score_field]), row.get("cache_reused") is False, row.get("evaluator_version"))
            if statuses[key] == "success":
                predictions[key] = values
            else:
                provisional_predictions[key] = values
        except (ValueError, KeyError):
            statuses[key] = "invalid_score"
            failed_predictions[run] += 1
    runs = sorted({run for _, run in statuses})
    if model_run is None:
        if len(runs) > 1:
            raise ValueError("Multiple model run_id values; choose --model-run to prevent cherry-picked averaging")
        model_run = runs[0] if runs else "legacy"
    if runs and model_run not in runs:
        raise ValueError(f"Requested model run does not exist: {model_run}")
    selected_predictions = {sample_id: values[0] for (sample_id, run), values in predictions.items() if run == model_run}
    selected_provisional = {sample_id: values[0] for (sample_id, run), values in provisional_predictions.items() if run == model_run}
    by_split = {}
    for split in ("development", "heldout"):
        members = [row for row in samples.values() if row["split"] == split]
        matched, provisional_matched = [], []
        multi_rater_matched = []
        inter_rater_errors, rater_pairs = [], defaultdict(list)
        for row in members:
            sid = row["sample_id"]
            raters = human_scores[sid]
            for (rater_a, a), (rater_b, b) in itertools.combinations(sorted(raters), 2):
                inter_rater_errors.append(abs(a - b))
                rater_pairs[(rater_a, rater_b)].append((a, b))
            if raters and sid in selected_predictions:
                item = (sid, row["pair_id"], statistics.mean(v for _, v in raters), selected_predictions[sid])
                matched.append(item)
                if len(raters) >= 2:
                    multi_rater_matched.append(item)
            if raters and sid in selected_provisional:
                provisional_matched.append((sid, row["pair_id"], statistics.mean(v for _, v in raters), selected_provisional[sid]))
        def model_metrics(items):
            return {
                "matched_samples": len(items),
                "mae": statistics.mean(abs(h - m) for _, _, h, m in items) if items else None,
                "mae_group_bootstrap_95pct": bootstrap_mae(items),
                "spearman": spearman([r[2] for r in items], [r[3] for r in items]),
                "within_original_pair_preference": preference_accuracy(items, True),
                "all_sample_pairs_preference": preference_accuracy(items, False),
            }
        by_split[split] = {
            "total_samples": len(members),
            "samples_with_independent_human_labels": sum(bool(human_scores[r["sample_id"]]) for r in members),
            "samples_with_two_or_more_raters": sum(len(human_scores[r["sample_id"]]) >= 2 for r in members),
            "model_scores_available": sum(r["sample_id"] in selected_predictions for r in members),
            "latest_model_status_counts": dict(Counter(statuses.get((r["sample_id"], model_run), "missing") for r in members)),
            "usable_score_coverage": sum(r["sample_id"] in selected_predictions for r in members) / len(members) if members else None,
            "all_labeled_model_comparison": model_metrics(matched),
            "two_or_more_rater_model_comparison": model_metrics(multi_rater_matched),
            "provisional_needs_review_comparison_diagnostic_only": model_metrics(provisional_matched),
            "inter_rater": {
                "pairwise_rating_comparisons": len(inter_rater_errors),
                "mae": statistics.mean(inter_rater_errors) if inter_rater_errors else None,
                "pairs": [{"raters": list(pair), "shared_samples": len(values),
                           "mae": statistics.mean(abs(a - b) for a, b in values),
                           "spearman": spearman([a for a, _ in values], [b for _, b in values])}
                          for pair, values in sorted(rater_pairs.items())],
            },
        }
    repeats = defaultdict(list)
    for (sid, run), (score, explicitly_uncached, version) in predictions.items():
        if explicitly_uncached and isinstance(version, str) and version:
            repeats[version, sid].append(score)
    repeat_sets = [values for values in repeats.values() if len(values) >= 2]
    heldout = by_split["heldout"]
    no_labels = heldout["samples_with_independent_human_labels"] == 0
    return {
        "manifest_sha256": manifest["manifest_sha256"], "rubric_version": RUBRIC_VERSION,
        "selected_model_run": model_run, "eligible_human_annotations": len(labels),
        "excluded_annotations": dict(excluded), "failed_or_invalid_prediction_records": dict(failed_predictions),
        "by_split": by_split,
        "repeatability": {
            "available_run_ids": runs, "sample_version_groups_with_at_least_two_uncached_runs": len(repeat_sets),
            "mean_within_sample_std": statistics.mean(statistics.pstdev(v) for v in repeat_sets) if repeat_sets else None,
            "mean_within_sample_range": statistics.mean(max(v) - min(v) for v in repeat_sets) if repeat_sets else None,
            "max_within_sample_range": max((max(v) - min(v) for v in repeat_sets), default=None),
            "note": "Requires distinct run_id values, explicit cache_reused=false, and matching nonempty evaluator_version (code/prompt/model/config identity). Unknown cache provenance and legacy records cannot establish repeatability. Different evaluator versions are never grouped together.",
        },
        "human_alignment_achieved": False,
        "evidence_status": "insufficient_independent_heldout_human_labels" if no_labels else "human_metrics_available_review_required",
        "claim_limit": "No independent held-out human labels: human alignment cannot be claimed." if no_labels else "These are measured agreements, not an automatic human-level certificate. Review held-out coverage, inter-rater reliability, uncertainty, failure rates, and a prospectively agreed acceptance target. Human provenance/independence is a rater declaration and must be checked operationally.",
        "metric_notes": [
            "Means over raters form per-video reference scores; model and human scores are not substituted for missing labels.",
            "Metrics describe this selected success subset, without population weighting or correction for unfinished evaluations.",
            "The MAE interval resamples original video-pair groups. Pairwise counts are dependent comparisons, not independent sample counts.",
            "A high score correlation can coexist with badly calibrated absolute rewards; inspect MAE and case errors together.",
            "Only status=success contributes to usable model agreement. Numeric needs_review scores are reported separately as provisional diagnostics; they do not establish usable alignment. Error/missing outcomes count against coverage.",
            "Compare model-versus-consensus and inter-rater MAE carefully: their reference targets differ.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--results", type=Path, required=True)
    freeze.add_argument("--reserve", type=Path, action="append", required=True, help="JSON object/list of previously inspected cases; repeat for cases.json and frame_count_case.json")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", default="20260910")
    freeze.add_argument("--heldout-fraction", type=float, default=.3)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--manifest", type=Path, required=True)
    metrics.add_argument("--labels", type=Path, required=True)
    metrics.add_argument("--predictions", type=Path, required=True)
    metrics.add_argument("--score-field", default="new_score")
    metrics.add_argument("--model-run")
    metrics.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_manifest(read_jsonl(args.results), reserved_sample_ids(args.reserve), args.seed, args.heldout_fraction)
        # A split cannot be overwritten after looking at results.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("manifest_sha256", "sample_count", "pair_count", "counts_by_split")}, ensure_ascii=False))
    else:
        result = evaluate(json.loads(args.manifest.read_text()), read_jsonl(args.labels), read_jsonl(args.predictions), args.score_field, args.model_run)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"evidence_status": result["evidence_status"], "human_alignment_achieved": False, "eligible_human_annotations": result["eligible_human_annotations"]}))


if __name__ == "__main__":
    main()
