"""Describe development evidence runs without substituting controls for human truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("samples", value.get("rows", [value]))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a list/JSONL of sample objects: {path}")
    return value


def valid_score(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def score_stats(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values, default=None), "max": max(values, default=None),
            "frequencies": dict(sorted(Counter(str(round(v, 8)) for v in values).items()))}


def token_value(usage, key):
    value = usage.get(key)
    return value if type(value) is int and value >= 0 else None


def trace_summary(entries):
    usage_entries = [entry for entry in entries if isinstance(entry.get("usage"), dict)]
    def usage_sum(selected):
        result = {key: sum(token_value(entry["usage"], key) or 0 for entry in selected)
                  for key in ("input_tokens", "output_tokens", "total_tokens")}
        result["input_prompt_cache_tokens"] = sum(
            token_value(entry["usage"].get("input_tokens_details") or {}, "cached_tokens") or 0
            for entry in selected)
        result["entries_with_usage"] = len(selected)
        result["entries_missing_input_or_output_tokens"] = sum(
            token_value(entry["usage"], "input_tokens") is None or token_value(entry["usage"], "output_tokens") is None
            for entry in selected)
        return result
    uncached = [entry for entry in usage_entries if entry.get("cache_reused") is False]
    cached = [entry for entry in usage_entries if entry.get("cache_reused") is True]
    unknown = [entry for entry in usage_entries if not isinstance(entry.get("cache_reused"), bool)]
    # A repeated response_id is the same API result, even if copied to two traces.
    unique = {}
    for index, entry in enumerate(uncached):
        key = entry.get("response_id") or ("trace-without-response-id", index)
        unique[key] = entry
    return {
        "trace_entries": len(entries), "by_stage": dict(Counter(entry.get("stage", "unknown") for entry in entries)),
        "schema_validation_rejections": sum(bool(entry.get("validation_error")) for entry in entries),
        "schema_retry_attempts": sum(type(entry.get("attempt")) is int and entry["attempt"] > 1 for entry in entries),
        "stage_error_entries": sum(bool(entry.get("error")) for entry in entries),
        "cache_hit_entries": sum(entry.get("cache_reused") is True for entry in entries),
        "recorded_usage_including_cache_hits": usage_sum(usage_entries),
        "known_uncached_unique_response_usage": usage_sum(list(unique.values())),
        "cache_reused_usage_not_new_api_consumption": usage_sum(cached),
        "unknown_cache_provenance_usage": usage_sum(unknown),
        "usage_limit": "Only recorded API response usage is summed. Cache-hit usage describes an earlier response; it is not new consumption. Transport-internal retries without recorded usage and missing error traces cannot be billed from this file.",
    }


def flatten_trace(trace, sample_id, record_index, source):
    if not isinstance(trace, list):
        return []
    entries = []
    for index, entry in enumerate(trace):
        if not isinstance(entry, dict):
            continue
        entries.append({"sample_id": sample_id, "record_index": record_index, "trace_index": index,
                        "trace_source": source, **{key: entry[key] for key in (
                            "stage", "attempt", "response_id", "usage", "cache_reused", "validation_error", "error") if key in entry}})
    return entries


def summarize(result_rows, manifest_rows, baseline_rows=None, trace_dir=None):
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if not manifest or len(manifest) != len(manifest_rows):
        raise ValueError("Manifest must contain unique, nonempty sample records")
    latest, history, ignored = {}, [], Counter()
    for index, row in enumerate(result_rows):
        sid = row.get("sample_id")
        if sid not in manifest:
            ignored["result_records_outside_manifest"] += 1
            continue
        latest[sid] = (index, row)
        history.extend(flatten_trace((row.get("details") or {}).get("trace"), sid, index, "result_record"))
    baseline = {row["sample_id"]: row for row in (baseline_rows or [])}
    samples, latest_traces, warnings = [], [], []
    for sid, spec in manifest.items():
        index, result = latest.get(sid, (None, {}))
        details = result.get("details") or {}
        status = result.get("status", "missing")
        raw_score = result.get("new_score")
        score = raw_score if valid_score(raw_score) else None
        review_required = details.get("review_required")
        eligible = status == "success" and result.get("training_eligible") is True and score is not None and review_required is not True
        if status == "success" and not eligible:
            warnings.append(f"{sid}: status=success but score/training_eligible/review_required are inconsistent")
        if raw_score is not None and score is None:
            warnings.append(f"{sid}: invalid numeric reward excluded from score statistics")
        report = details.get("evidence_report") or {}
        task = report.get("task_assessment") or {}
        physics = report.get("physics_assessment") or {}
        visual = report.get("visual_assessment") or {}
        old = baseline.get(sid) or {}
        old_score = old.get("new_score") if old.get("status") == "success" else None
        if not valid_score(old_score):
            old_score = result.get("baseline_doubao_score", spec.get("new_score"))
        old_score = old_score if valid_score(old_score) else None
        trace = flatten_trace(details.get("trace"), sid, index, "result_record")
        if not trace and trace_dir is not None and result:
            identity = hashlib.sha256((str(spec["video_path"]) + "\n" + spec["prompt"]).encode()).hexdigest()[:20]
            sidecar = trace_dir / (identity + ".json")
            if sidecar.is_file():
                value = json.loads(sidecar.read_text())
                if (value.get("video_path") == spec["video_path"] and value.get("prompt") == spec["prompt"]
                        and value.get("evaluator_version") == result.get("evaluator_version")
                        and result.get("evaluator_version")):
                    trace = flatten_trace(value.get("trace"), sid, index, "persistent_sidecar_latest_attempt")
                    history.extend(trace)
                else:
                    warnings.append(f"{sid}: ignored sidecar with mismatched source identity/evaluator version")
        latest_traces.extend(trace)
        synthetic = spec.get("synthetic_control") is True
        expected = spec.get("expected_task_verdict") if synthetic else None
        actual = task.get("verdict")
        issues = physics.get("issues") or []
        samples.append({
            "sample_id": sid, "audit_id": spec.get("audit_id"), "synthetic_control": synthetic,
            "status": status, "training_eligible": eligible,
            "new_score": score, "baseline_doubao_score": old_score,
            "numeric_delta_from_baseline": score - old_score if score is not None and old_score is not None else None,
            "task_verdict": actual, "task_confidence": task.get("confidence"), "target_match": task.get("target_match"),
            "physics_verdict": physics.get("verdict"), "physics_confidence": physics.get("confidence"),
            "visual_verdict": visual.get("verdict"),
            "physics_issues": [{key: issue.get(key) for key in ("kind", "severity", "certainty", "evidence", "reason", "alternative_explanation")}
                               for issue in issues if isinstance(issue, dict)],
            "task_reason": task.get("reason"), "physics_reason": physics.get("reason"),
            "review_reasons": (details.get("scoring") or {}).get("review_reasons", []),
            "control_source_id": spec.get("control_source_id"), "expected_task_verdict": expected,
            "expectation_basis": spec.get("expectation_basis"),
            "construction": spec.get("construction"), "expected_observation": spec.get("expected_observation"),
            "control_claim_limit": spec.get("claim_limit") if synthetic else None,
            "control_task_failure_identified": actual == expected if expected is not None and actual is not None else None,
            "control_failure_identification_is_usable": eligible and actual == expected if expected is not None else None,
            "run_id": result.get("run_id"), "evaluator_version": result.get("evaluator_version"),
            "cache_reused": result.get("cache_reused"), "elapsed_seconds": result.get("elapsed_seconds"),
            "error": result.get("error"), "trace_summary": trace_summary(trace),
        })

    def category_summary(selected):
        counts = Counter(row["status"] for row in selected)
        eligible = [row for row in selected if row["training_eligible"]]
        provisional = [row for row in selected if row["status"] == "needs_review"]
        numeric_provisional = [row["new_score"] for row in provisional if row["new_score"] is not None]
        def conclusions(rows):
            return {key: dict(Counter(row[key] or "not_assessed" for row in rows)) for key in
                    ("task_verdict", "target_match", "physics_verdict", "visual_verdict")}
        comparable = [row for row in eligible if row["baseline_doubao_score"] is not None]
        provisional_comparable = [row for row in provisional if row["baseline_doubao_score"] is not None and row["new_score"] is not None]
        issues = Counter((issue["kind"], issue["severity"], issue["certainty"])
                         for row in selected for issue in row["physics_issues"])
        return {
            "expected_samples": len(selected), "latest_status_counts": dict(counts),
            "attempted_samples": len(selected) - counts["missing"],
            "usable_training_samples": len(eligible), "usable_training_coverage": len(eligible) / len(selected) if selected else None,
            "needs_review_samples": len(provisional), "needs_review_numeric_samples": len(numeric_provisional),
            "needs_review_without_numeric_score": len(provisional) - len(numeric_provisional),
            "usable_scores": score_stats([row["new_score"] for row in eligible]),
            "provisional_scores_not_training_rewards": score_stats(numeric_provisional),
            "matched_usable_baseline_scores": score_stats([row["baseline_doubao_score"] for row in comparable]),
            "matched_usable_new_scores": score_stats([row["new_score"] for row in comparable]),
            "matched_usable_delta": score_stats([row["numeric_delta_from_baseline"] for row in comparable]),
            "provisional_numeric_delta_diagnostic_only": score_stats([row["numeric_delta_from_baseline"] for row in provisional_comparable]),
            "conclusions_all_returned_statuses": conclusions(selected),
            "conclusions_usable_only": conclusions(eligible),
            "physics_issue_counts": [{"kind": kind, "severity": severity, "certainty": certainty, "count": count}
                                     for (kind, severity, certainty), count in sorted(issues.items())],
            "samples_without_recorded_trace": sum(row["status"] != "missing" and row["trace_summary"]["trace_entries"] == 0 for row in selected),
        }

    real = [row for row in samples if not row["synthetic_control"]]
    controls = [row for row in samples if row["synthetic_control"]]
    pairs = defaultdict(list)
    for row in real:
        pairs[row["sample_id"].rpartition("/")[0]].append(row)
    pair_changes = []
    for pair_id, members in sorted(pairs.items()):
        if len(members) != 2 or len({manifest[row["sample_id"]]["prompt"] for row in members}) != 1:
            continue
        members.sort(key=lambda row: row["sample_id"])
        a, b = members
        old_delta = a["baseline_doubao_score"] - b["baseline_doubao_score"] if all(row["baseline_doubao_score"] is not None for row in members) else None
        new_delta = a["new_score"] - b["new_score"] if all(row["new_score"] is not None for row in members) else None
        pair_changes.append({"pair_id": pair_id, "sample_ids_in_difference_order": [row["sample_id"] for row in members],
                             "statuses": [row["status"] for row in members], "baseline_score_difference": old_delta,
                             "new_numeric_difference": new_delta, "both_new_scores_training_eligible": all(row["training_eligible"] for row in members),
                             "interpretation": "Numeric changes alone do not establish which candidate humans prefer; needs_review values are provisional."})
    controls_summary = category_summary(controls)
    explicit_controls = [row for row in controls if row["expected_task_verdict"] is not None]
    returned_controls = [row for row in controls if row["task_verdict"] is not None]
    assessed_controls = [row for row in explicit_controls if row["task_verdict"] is not None]
    controls_summary["constructed_task_failure_checks"] = {
        "total_controls": len(controls), "expected_controls": len(explicit_controls),
        "controls_with_explicit_task_expectation": len(explicit_controls),
        "with_returned_task_verdict": len(returned_controls),
        "explicit_expectations_with_returned_task_verdict": len(assessed_controls),
        "controls_without_unique_task_label": len(controls) - len(explicit_controls),
        "matched_expectation_all_returned": sum(row["control_task_failure_identified"] is True for row in assessed_controls),
        "matched_expectation_and_training_eligible": sum(row["control_failure_identification_is_usable"] is True for row in controls),
        "interpretation": "Explicit expected_task_verdict values are compared only when provided. Other diagnostic controls preserve construction/expected_observation without inventing a unique task or physics label or automatic match result. None are human labels or an estimate of natural-video accuracy.",
    }
    return {
        "schema_version": 1, "expected_samples": len(samples), "all_expected_have_results": len(latest) == len(samples),
        "ignored_records": dict(ignored), "run_ids": sorted({row["run_id"] for row in samples if row["run_id"]}),
        "evaluator_versions": sorted({row["evaluator_version"] for row in samples if row["evaluator_version"]}),
        "real_cases": category_summary(real), "synthetic_controls": controls_summary,
        "same_task_original_pair_changes": pair_changes,
        "latest_attempt_trace_summary": trace_summary(latest_traces),
        "all_recorded_attempts_trace_summary": trace_summary(history),
        "per_trace_records": history, "samples": samples, "warnings": warnings,
        "human_accuracy_established": False,
        "claim_limit": "No independent human annotations are read by this report. It evaluates output behavior, evidence consistency diagnostics, coverage and constructed controls only. Higher scores or reduced pair gaps do not prove improved human alignment.",
    }


def fmt(value):
    return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def safe(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def markdown_report(stats):
    real, controls = stats["real_cases"], stats["synthetic_controls"]
    lines = ["# Reward agent 开发对照评测", "",
             "本报告区分可训练评分、待复核暂定评分和失败。真实视频与人为构造的负对照分别统计；没有独立人工标签，因此不能声称达到人类准确度。", "",
             f"预期 {stats['expected_samples']} 个案例；全部已有结果：{'是' if stats['all_expected_have_results'] else '否，仍有缺失'}。",
             f"评测版本：{', '.join(stats['evaluator_versions']) or '未记录'}。", "",
             "## 覆盖率与分数", "",
             "| 集合 | 预期数 | 可训练 | 可训练覆盖率 | 待复核（有暂定数字 / 无数字） | 错误 | 缺失 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, section in (("真实 case", real), ("构造负对照", controls)):
        counts = section["latest_status_counts"]
        coverage = section["usable_training_coverage"]
        lines.append(f"| {label} | {section['expected_samples']} | {section['usable_training_samples']} | {coverage:.1%} | {section['needs_review_numeric_samples']} / {section['needs_review_without_numeric_score']} | {counts.get('error', 0)} | {counts.get('missing', 0)} |" if coverage is not None else f"| {label} | 0 | 0 | — | 0 / 0 | 0 | 0 |")
    lines += ["", "`success` 还需满足 training_eligible=true、有限数字分数及没有 review_required 才计入可训练覆盖率。API 返回成功不等于人工判断正确。", "",
              f"真实 case 可训练 reward 均值：{fmt(real['usable_scores']['mean'])}（n={real['usable_scores']['n']}）；待复核数字均值：{fmt(real['provisional_scores_not_training_rewards']['mean'])}（n={real['provisional_scores_not_training_rewards']['n']}，不作为可训练 reward）。",
              f"与旧豆包可配对且新评分可训练的子集：旧均值 {fmt(real['matched_usable_baseline_scores']['mean'])}，新均值 {fmt(real['matched_usable_new_scores']['mean'])}，n={real['matched_usable_new_scores']['n']}。分数改变本身不证明更准确。", "",
              "## 逐案例结果", "", "| 案例 | 类型 | 状态 | 旧豆包 | 新数字 | 可训练 | 任务结论 | 目标 | 物理结论 |", "|---|---|---|---:|---:|---|---|---|---|"]
    for row in stats["samples"]:
        lines.append("| " + " | ".join(map(safe, [row["audit_id"] or row["sample_id"], "构造" if row["synthetic_control"] else "真实", row["status"], fmt(row["baseline_doubao_score"]), fmt(row["new_score"]), "是" if row["training_eligible"] else "否", row["task_verdict"], row["target_match"], row["physics_verdict"]])) + " |")
    lines += ["", "## 同任务候选对", ""]
    if not stats["same_task_original_pair_changes"]:
        lines += ["本批没有两条均在 manifest 中的同任务原始候选对。"]
    for pair in stats["same_task_original_pair_changes"]:
        lines.append(f"- `{pair['pair_id']}`：按 {', '.join(pair['sample_ids_in_difference_order'])} 顺序相减，旧分差 {fmt(pair['baseline_score_difference'])}，新数字分差 {fmt(pair['new_numeric_difference'])}；两条均可训练：{'是' if pair['both_new_scores_training_eligible'] else '否'}。")
    lines += ["", "分差缩小或排序改变不能证明符合人类偏好；待复核数字只作诊断。", "", "## 任务与物理结论计数", ""]
    for label, section in (("真实 case", real), ("构造负对照", controls)):
        conclusions = section["conclusions_all_returned_statuses"]
        lines.append(f"- {label}：任务 {json.dumps(conclusions['task_verdict'], ensure_ascii=False)}；物理 {json.dumps(conclusions['physics_verdict'], ensure_ascii=False)}。not_assessed 包含错误或缺失，没有被当成任务失败。")
    checks = controls["constructed_task_failure_checks"]
    lines += ["", "## 构造对照", "",
              f"共 {checks['total_controls']} 条构造对照，{checks['with_returned_task_verdict']} 条实际返回任务类别；其中 {checks['controls_with_explicit_task_expectation']} 条预先设置明确任务类别期望，{checks['explicit_expectations_with_returned_task_verdict']} 条同时有可比较的返回类别。",
              f"仅在这些明确类别期望中，{checks['matched_expectation_all_returned']} 条类别匹配，{checks['matched_expectation_and_training_eligible']} 条同时满足可训练条件。待复核结论不等于稳定识别。",
              f"另有 {checks['controls_without_unique_task_label']} 条未设唯一任务类别标签，依据 construction / expected_observation 作诊断描述，不自动计为匹配或不匹配，也不推导唯一物理标签。",
              "这些构造操作及观察要求不是人工标注准确率；即使明确类别对照全部匹配，也不能证明自然视频准确度。", ""]
    for row in stats["samples"]:
        if row["synthetic_control"]:
            if row["expected_task_verdict"] is not None:
                lines.append(f"- `{row['sample_id']}`：明确任务期望 {safe(row['expected_task_verdict'])}，实际 {safe(row['task_verdict'])}，状态 {row['status']}。构造依据：{safe(row['expectation_basis'] or row['construction'])}")
            else:
                lines.append(f"- `{row['sample_id']}`：未设唯一任务/物理类别标签；实际任务 {safe(row['task_verdict'])}，物理 {safe(row['physics_verdict'])}，状态 {row['status']}。构造：{safe(row['construction'])} 观察要求：{safe(row['expected_observation'])} 不自动计算类别匹配。")
                if row["control_claim_limit"]:
                    lines.append(f"  范围限定：{safe(row['control_claim_limit'])}")
    trace = stats["all_recorded_attempts_trace_summary"]
    used = trace["known_uncached_unique_response_usage"]
    cached = trace["cache_reused_usage_not_new_api_consumption"]
    lines += ["", "## API 轨迹与重试", "",
              f"已记录 {trace['trace_entries']} 条阶段轨迹；结构/证据校验拒绝 {trace['schema_validation_rejections']} 次，schema 后续尝试 {trace['schema_retry_attempts']} 次，阶段错误条目 {trace['stage_error_entries']} 条；缓存命中 {trace['cache_hit_entries']} 条。",
              f"已知未复用且按 response_id 去重的 usage：输入 {used['input_tokens']}，输出 {used['output_tokens']}，总计 {used['total_tokens']} tokens。",
              f"缓存命中记录所携带的旧 usage：输入 {cached['input_tokens']}，输出 {cached['output_tokens']}；不算作本轮新增 API 消耗。",
              f"已有结果但没有可读 trace 的样本：真实 {real['samples_without_recorded_trace']}，构造 {controls['samples_without_recorded_trace']}。缺失轨迹或传输层内部失败可能使 token 统计不完整，以上不是账单金额。",
              "逐 trace 的输入、输出、缓存状态、阶段、尝试号、response_id 和校验错误保存在 stats.json 的 per_trace_records 中。", "", "## 需要复核的条目", ""]
    unresolved = [row for row in stats["samples"] if not row["training_eligible"]]
    if not unresolved:
        lines.append("本批无结构化待复核或调用错误；仍需独立人工核对事实和 reward 标度。")
    for row in unresolved:
        reasons = row["review_reasons"] or [row["error"] or "缺失结果或训练资格不一致"]
        lines.append(f"- `{row['sample_id']}`（{row['status']}）：{safe('; '.join(reasons))}")
    if stats["warnings"]:
        lines += ["", "数据一致性提醒："] + ["- " + safe(warning) for warning in stats["warnings"]]
    lines += ["", "本报告未读取任何独立人工标签，不给出人类准确率或“达到人类水平”的结论。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="Optional matching persistent traces, useful when an error result contains no details")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(read_rows(args.results), read_rows(args.manifest), read_rows(args.baseline) if args.baseline else None, args.trace_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "REVIEW.md").write_text(markdown_report(result))
    print(json.dumps({"expected": result["expected_samples"], "real_status": result["real_cases"]["latest_status_counts"], "control_status": result["synthetic_controls"]["latest_status_counts"], "human_accuracy_established": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
