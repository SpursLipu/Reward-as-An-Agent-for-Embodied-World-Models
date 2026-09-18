"""Compare two evidence runs; independent repeatability requires verified provenance.

Standard library only. Uses the shared JSON/JSONL reader in summarize_evidence.py.
No annotations are generated and none of these differences measure human accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts.evidence.summarize_evidence import read_rows, valid_score, safe, fmt


def normalized(record):
    record = record or {}
    details = record.get("details") or {}
    evidence = details.get("evidence_report") or {}
    status = record.get("status", "missing")
    score = record.get("new_score")
    score = score if valid_score(score) else None
    trace = [item for item in details.get("trace", []) if isinstance(item, dict)]
    task = evidence.get("task_assessment") or {}
    physics = evidence.get("physics_assessment") or {}
    return {
        "status": status, "score": score,
        "training_eligible": (status == "success" and record.get("training_eligible") is True
                              and score is not None and details.get("review_required") is not True),
        "task_verdict": task.get("verdict"), "target_match": task.get("target_match"),
        "physics_verdict": physics.get("verdict"), "task_confidence": task.get("confidence"),
        "physics_confidence": physics.get("confidence"),
        "review_reasons": (details.get("scoring") or {}).get("review_reasons", []),
        "evaluator_version": record.get("evaluator_version"), "run_id": record.get("run_id"),
        "cache_reused": record.get("cache_reused"),
        "trace_cache_hit_count": sum(item.get("cache_reused") is True for item in trace),
        "schema_validation_errors": sum(bool(item.get("validation_error")) for item in trace),
        "schema_retry_attempts": sum(type(item.get("attempt")) is int and item["attempt"] > 1 for item in trace),
        "trace_entries": len(trace), "error": record.get("error"),
    }


def independent_reasons(first, second, spec):
    reasons = []
    if not first or not second:
        return ["至少一轮缺失结果"]
    a, b = normalized(first), normalized(second)
    if not isinstance(a["evaluator_version"], str) or not a["evaluator_version"] or a["evaluator_version"] != b["evaluator_version"]:
        reasons.append("evaluator_version 不同或缺失")
    if (not isinstance(a["run_id"], str) or not a["run_id"] or not isinstance(b["run_id"], str)
            or not b["run_id"] or a["run_id"] == b["run_id"]):
        reasons.append("run_id 相同或缺失，不能证实两次独立执行")
    for label, original, item in (("第一轮", first, a), ("第二轮", second, b)):
        if item["cache_reused"] is not False:
            reasons.append(label + " cache_reused 未明确为 false")
        if item["trace_cache_hit_count"]:
            reasons.append(label + " trace 存在缓存命中")
        if any(original.get(key) != spec.get(key) or key not in original for key in ("video_path", "prompt")):
            reasons.append(label + "输入路径/任务与 manifest 不一致或缺失")
    return reasons


def numeric_differences(rows):
    values = [abs(row["second"]["score"] - row["first"]["score"]) for row in rows]
    changed = sum(value > .1 + 1e-9 for value in values)
    return {
        "sample_count": len(values),
        "mean_absolute_difference": statistics.mean(values) if values else None,
        "max_absolute_difference": max(values, default=None),
        "count_absolute_difference_gt_0_1": changed,
        "fraction_absolute_difference_gt_0_1": changed / len(values) if values else None,
        "threshold": .1,
        "interpretation": "Model-versus-model score differences on two training-eligible outputs; not human-label error.",
    }


def agreement(rows, field):
    available = [row for row in rows if row["first"][field] is not None and row["second"][field] is not None]
    same = sum(row["first"][field] == row["second"][field] for row in available)
    return {"same_count": same, "compared_samples": len(available),
            "agreement_fraction": same / len(available) if available else None,
            "not_compared_missing_verdict": len(rows) - len(available)}


def schema_frequency(records, members):
    relevant = [row for row in records if row.get("sample_id") in members]
    traces = []
    missing_trace = 0
    affected_ids = set()
    for row in relevant:
        trace = [item for item in (row.get("details") or {}).get("trace", []) if isinstance(item, dict)]
        if not trace:
            missing_trace += 1
        traces.extend(trace)
        if any(item.get("validation_error") for item in trace):
            affected_ids.add(row["sample_id"])
    # Error-only transport entries did not return a JSON object to validate.
    validated_outputs = [item for item in traces if item.get("response_id") or item.get("usage") is not None or "output" in item or item.get("validation_error")]
    errors = sum(bool(item.get("validation_error")) for item in validated_outputs)
    attempted_ids = {row["sample_id"] for row in relevant}
    return {
        "result_attempt_records": len(relevant), "attempted_samples": len(attempted_ids),
        "recorded_model_output_attempts": len(validated_outputs), "schema_validation_errors": errors,
        "schema_error_fraction_per_recorded_output": errors / len(validated_outputs) if validated_outputs else None,
        "samples_with_schema_error": len(affected_ids),
        "schema_affected_fraction_of_attempted_samples": len(affected_ids) / len(attempted_ids) if attempted_ids else None,
        "schema_retry_attempts": sum(type(item.get("attempt")) is int and item["attempt"] > 1 for item in traces),
        "result_records_without_trace": missing_trace,
        "note": "Includes recorded retries, not just final outputs. Error results with no trace can undercount validation failures; no missing error is invented.",
    }


def compare(first_rows, second_rows, manifest_rows):
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if not manifest or len(manifest) != len(manifest_rows):
        raise ValueError("Manifest must contain unique sample IDs")
    first = {row["sample_id"]: row for row in first_rows if row.get("sample_id") in manifest}
    second = {row["sample_id"]: row for row in second_rows if row.get("sample_id") in manifest}
    rows, warnings = [], []
    for sid, spec in manifest.items():
        a, b = normalized(first.get(sid)), normalized(second.get(sid))
        reasons = independent_reasons(first.get(sid), second.get(sid), spec)
        same_version = bool(a["evaluator_version"]) and a["evaluator_version"] == b["evaluator_version"]
        both_eligible = a["training_eligible"] and b["training_eligible"]
        delta = b["score"] - a["score"] if a["score"] is not None and b["score"] is not None else None
        row = {
            "sample_id": sid, "audit_id": spec.get("audit_id"),
            "synthetic_control": spec.get("synthetic_control") is True,
            "expected_task_verdict": spec.get("expected_task_verdict") if spec.get("synthetic_control") is True else None,
            "first": a, "second": b,
            "same_evaluator_version": same_version,
            "independent_repeat_provenance_verified": not reasons,
            "excluded_from_independent_repeat_reasons": reasons,
            "both_training_eligible": both_eligible,
            "numeric_difference_second_minus_first": delta,
            "numeric_difference_is_provisional_or_ineligible": delta is not None and not both_eligible,
            "status_changed": a["status"] != b["status"],
            "eligibility_changed": a["training_eligible"] != b["training_eligible"],
            "verdict_changes": {field: (a[field] != b[field]) if a[field] is not None and b[field] is not None else None
                                for field in ("task_verdict", "target_match", "physics_verdict")},
        }
        for name, item in (("first", a), ("second", b)):
            if item["status"] == "success" and not item["training_eligible"]:
                warnings.append(f"{sid} {name}: success is not consistently training-eligible")
        rows.append(row)

    def category(selected):
        eligible = [row for row in selected if row["both_training_eligible"]]
        independent = [row for row in selected if row["independent_repeat_provenance_verified"]]
        independent_eligible = [row for row in independent if row["both_training_eligible"]]
        coverage = {}
        for name in ("first", "second"):
            available = [row for row in selected if row[name]["status"] != "missing"]
            n_eligible = sum(row[name]["training_eligible"] for row in selected)
            coverage[name] = {
                "expected_samples": len(selected), "samples_with_results": len(available),
                "status_counts": dict(Counter(row[name]["status"] for row in selected)),
                "training_eligible_samples": n_eligible,
                "training_eligible_coverage": n_eligible / len(selected) if selected else None,
            }
        coverage["training_eligible_count_change_second_minus_first"] = coverage["second"]["training_eligible_samples"] - coverage["first"]["training_eligible_samples"]
        coverage["training_eligible_coverage_change_second_minus_first"] = (coverage["second"]["training_eligible_coverage"] - coverage["first"]["training_eligible_coverage"]) if selected else None
        return {
            "coverage": coverage,
            "status_transitions": dict(Counter(row["first"]["status"] + " -> " + row["second"]["status"] for row in selected)),
            "eligibility_changes": sum(row["eligibility_changed"] for row in selected),
            "independent_repeat_provenance_samples": len(independent),
            "independent_repeat_exclusion_reasons": dict(Counter(reason for row in selected for reason in row["excluded_from_independent_repeat_reasons"])),
            "independent_repeatability_common_training_eligible": {
                "numeric": numeric_differences(independent_eligible),
                "task_verdict": agreement(independent_eligible, "task_verdict"),
                "target_match": agreement(independent_eligible, "target_match"),
                "physics_verdict": agreement(independent_eligible, "physics_verdict"),
            },
            "independent_returned_verdicts_including_needs_review_diagnostic_only": {
                "task_verdict": agreement(independent, "task_verdict"),
                "target_match": agreement(independent, "target_match"),
                "physics_verdict": agreement(independent, "physics_verdict"),
            },
            "descriptive_common_training_eligible_not_necessarily_independent": {
                "numeric": numeric_differences(eligible), "task_verdict": agreement(eligible, "task_verdict"),
                "target_match": agreement(eligible, "target_match"), "physics_verdict": agreement(eligible, "physics_verdict"),
            },
            "descriptive_all_returned_verdicts_including_provisional": {
                "task_verdict": agreement(selected, "task_verdict"), "target_match": agreement(selected, "target_match"),
                "physics_verdict": agreement(selected, "physics_verdict"),
            },
            "schema_frequency": {
                "first": schema_frequency(first_rows, {row["sample_id"] for row in selected}),
                "second": schema_frequency(second_rows, {row["sample_id"] for row in selected}),
            },
        }

    real = [row for row in rows if not row["synthetic_control"]]
    controls = [row for row in rows if row["synthetic_control"]]
    independent_count = sum(row["independent_repeat_provenance_verified"] for row in rows)
    return {
        "schema_version": 1, "expected_samples": len(rows),
        "comparison_kind": "independent_repeat_subset_available" if independent_count else "descriptive_comparison_only",
        "independent_repeat_provenance_samples": independent_count,
        "metadata": {
            "first_evaluator_versions": sorted({row["first"]["evaluator_version"] for row in rows if row["first"]["evaluator_version"]}),
            "second_evaluator_versions": sorted({row["second"]["evaluator_version"] for row in rows if row["second"]["evaluator_version"]}),
            "first_run_ids": sorted({row["first"]["run_id"] for row in rows if row["first"]["run_id"]}),
            "second_run_ids": sorted({row["second"]["run_id"] for row in rows if row["second"]["run_id"]}),
        },
        "ignored_first_result_records": sum(row.get("sample_id") not in manifest for row in first_rows),
        "ignored_second_result_records": sum(row.get("sample_id") not in manifest for row in second_rows),
        "real_cases": category(real), "synthetic_controls": category(controls),
        "cases": rows, "warnings": warnings,
        "human_accuracy_established": False,
        "interpretation_limits": [
            "Independent repeated execution requires the same evaluator_version, distinct nonempty run IDs, explicit cache_reused=false in both rounds, no cache-hit trace entry, and matching manifest input path/task.",
            "Different versions, cached results or missing provenance are described separately; they are never reported as independent repeatability.",
            "Numeric repeatability includes only outputs that are training-eligible in both rounds. Provisional needs_review scores are shown per case but are excluded from those aggregates.",
            "The >0.1 statistic uses an absolute reward difference strictly greater than 0.1 with 1e-9 floating-point tolerance; every agreement reports its own denominator.",
            "Manifest path/task equality does not authenticate source-video byte immutability or provenance declarations; retain immutable inputs and run configs operationally.",
            "Repeatability and constructed-control behavior do not establish accuracy or similarity to human judgment. No human labels are created or read.",
        ],
    }


def agreement_text(item):
    return f"{item['same_count']}/{item['compared_samples']}（{item['agreement_fraction']:.1%}）" if item["compared_samples"] else "—（分母 0）"


def render(stats):
    independent_title = ("同版本独立重测：两轮共同可训练子集" if stats["independent_repeat_provenance_samples"]
                         else "来源核验：本次只能进行描述性对照，独立重复性统计为空")
    lines = ["# 两轮 reward 评测对照", "",
             f"预期 {stats['expected_samples']} 条，逐样本满足同版本、不同 run_id、两轮明确未复用缓存及输入一致的记录有 **{stats['independent_repeat_provenance_samples']} 条**。",
             "其余仅作描述性对照。共同可训练结果、待复核暂定结果和状态变化分别报告；没有人工标签，以下不是准确率。", "",
             f"第一轮版本：{', '.join(stats['metadata']['first_evaluator_versions']) or '缺失'}。",
             f"第二轮版本：{', '.join(stats['metadata']['second_evaluator_versions']) or '缺失'}。", "",
             "## 有效覆盖率", "",
             "| 集合 | 预期数 | 第一轮可训练 | 第二轮可训练 | 覆盖率变化 | 状态转换 |",
             "|---|---:|---:|---:|---:|---|"]
    for label, key in (("真实 case", "real_cases"), ("构造对照", "synthetic_controls")):
        section = stats[key]
        coverage = section["coverage"]
        a, b = coverage["first"], coverage["second"]
        change = coverage["training_eligible_coverage_change_second_minus_first"]
        change_text = f"{change:+.1%}" if change is not None else "—"
        lines.append(f"| {label} | {a['expected_samples']} | {a['training_eligible_samples']} | {b['training_eligible_samples']} | {change_text} | {safe(json.dumps(section['status_transitions'], ensure_ascii=False))} |")
    lines += ["", "分母包含 manifest 中全部预期样本，失败、待复核和缺失都不会被静默丢掉或当作 0 分。", "", "## " + independent_title, "",
              "| 集合 | 数字分差分母 | 平均绝对分差 | 最大绝对分差 | 分差 >0.1 | 任务结论一致 | 目标结论一致 | 物理结论一致 |",
              "|---|---:|---:|---:|---:|---|---|---|"]
    for label, key in (("真实 case", "real_cases"), ("构造对照", "synthetic_controls")):
        section = stats[key]["independent_repeatability_common_training_eligible"]
        numeric = section["numeric"]
        threshold = f"{numeric['count_absolute_difference_gt_0_1']}/{numeric['sample_count']}" if numeric["sample_count"] else "—（分母 0）"
        if numeric["sample_count"]:
            threshold += f"（{numeric['fraction_absolute_difference_gt_0_1']:.1%}）"
        lines.append(f"| {label} | {numeric['sample_count']} | {fmt(numeric['mean_absolute_difference'])} | {fmt(numeric['max_absolute_difference'])} | {threshold} | {agreement_text(section['task_verdict'])} | {agreement_text(section['target_match'])} | {agreement_text(section['physics_verdict'])} |")
    lines += ["", "上述差值比较的是模型两次输出，不是模型相对人类标签的误差。构造对照单独列出，不与真实视频合成一个稳定性指标。", "", "## 待复核状态与全部返回结论", ""]
    for label, key in (("真实 case", "real_cases"), ("构造对照", "synthetic_controls")):
        section = stats[key]
        item = section["independent_returned_verdicts_including_needs_review_diagnostic_only"]
        if section["independent_repeat_provenance_samples"]:
            lines.append(f"- {label}：同版本、无缓存重测中，包含待复核输出的任务结论一致 {agreement_text(item['task_verdict'])}；目标一致 {agreement_text(item['target_match'])}；物理一致 {agreement_text(item['physics_verdict'])}。这里含暂定判断，仅作诊断。")
        else:
            lines.append(f"- {label}：没有满足独立重复性条件的样本，只报告描述性对照。")
        descriptive = section["descriptive_common_training_eligible_not_necessarily_independent"]
        lines.append(f"- {label}：不要求版本/缓存独立性的共同可训练描述性对照 n={descriptive['numeric']['sample_count']}，平均绝对分差 {fmt(descriptive['numeric']['mean_absolute_difference'])}；该数不能称为独立重测稳定性。")
    lines += ["", "## 逐案例变化", "", "| 案例 | 类型 | 第一轮状态/分数 | 第二轮状态/分数 | 分差（第二减第一） | 独立且共同可训练 | 任务变化 | 目标变化 | 物理变化 |",
              "|---|---|---|---|---:|---|---|---|---|"]
    for row in stats["cases"]:
        a, b = row["first"], row["second"]
        change = fmt(row["numeric_difference_second_minus_first"])
        if row["numeric_difference_is_provisional_or_ineligible"]:
            change += "（暂定）"
        parts = [row["audit_id"] or row["sample_id"], "构造" if row["synthetic_control"] else "真实",
                 f"{a['status']} / {fmt(a['score'])}", f"{b['status']} / {fmt(b['score'])}", change,
                 "是" if row["independent_repeat_provenance_verified"] and row["both_training_eligible"] else "否"]
        parts += [f"{a[field] or '未评'} → {b[field] or '未评'}" for field in ("task_verdict", "target_match", "physics_verdict")]
        lines.append("| " + " | ".join(safe(part) for part in parts) + " |")
    exclusions = [row for row in stats["cases"] if row["excluded_from_independent_repeat_reasons"]]
    if exclusions:
        lines += ["", "不能计入独立重测的记录：", ""]
        lines += [f"- `{row['sample_id']}`：{'；'.join(row['excluded_from_independent_repeat_reasons'])}。" for row in exclusions]
    lines += ["", "## Schema 校验错误频率", "", "| 集合/轮次 | 被记录的模型输出次数 | 校验错误次数 | 校验错误比例 | 涉及样本 / 已尝试样本 | 无 trace 的结果记录 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("真实", "real_cases"), ("构造", "synthetic_controls")):
        for round_name, round_label in (("first", "第一轮"), ("second", "第二轮")):
            item = stats[key]["schema_frequency"][round_name]
            fraction = item["schema_error_fraction_per_recorded_output"]
            lines.append(f"| {label}/{round_label} | {item['recorded_model_output_attempts']} | {item['schema_validation_errors']} | {fraction:.1%} | {item['samples_with_schema_error']}/{item['attempted_samples']} | {item['result_records_without_trace']} |" if fraction is not None else f"| {label}/{round_label} | 0 | {item['schema_validation_errors']} | — | {item['samples_with_schema_error']}/{item['attempted_samples']} | {item['result_records_without_trace']} |")
    lines += ["", "校验错误统计包含已有 trace 中的结构/证据重试，未记录 trace 的调用失败可能使频率低估。Schema 通过只说明结构满足约束，不代表事实正确。"]
    if stats["warnings"]:
        lines += ["", "数据一致性提醒：", ""] + ["- " + safe(item) for item in stats["warnings"]]
    lines += ["", "本报告没有生成或读取人工标签；稳定性、负对照通过及分数变化都不能替代人与模型一致性验证。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_rows(args.first), read_rows(args.second), read_rows(args.manifest))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "COMPARISON.md").write_text(render(result))
    print(json.dumps({"comparison_kind": result["comparison_kind"], "independent_repeat_provenance_samples": result["independent_repeat_provenance_samples"], "expected_samples": result["expected_samples"], "human_accuracy_established": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
