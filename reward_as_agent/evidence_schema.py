"""Evidence report validation and a provisional, deterministic reward rubric.

The model supplies categorical judgements and inspectable evidence, never numbers.
Validation catches structural and categorical contradictions; it does not pretend
to resolve contradictions in natural-language reasons. Those require visual review.
The reducer is deliberately not described as human-calibrated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


SCHEMA_VERSION = "evidence-v2"
SCORING_VERSION = "evidence-soft-physics-v2-provisional"
TASK_SCORES = {"complete": 1.0, "mostly_complete": 0.8, "partial": 0.45, "failed": 0.0}
PHYSICS_SCORES = {"plausible": 1.0, "minor_defect": 0.75, "major_defect": 0.25}
VISUAL_SCORES = {"clear": 1.0, "minor_degradation": 0.8, "severe_degradation": 0.3}


class EvidenceValidationError(ValueError):
    """A report cannot be safely reduced to a reward."""


def _fail(path: str, message: str) -> None:
    raise EvidenceValidationError(f"{path}: {message}")


def _object(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be a JSON object")
    missing = keys - value.keys()
    if missing:
        _fail(path, f"missing required fields {sorted(missing)!r}")
    extra = value.keys() - keys
    if extra:
        _fail(path, f"unknown fields {sorted(extra, key=str)!r}; numeric model scores are not accepted")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a nonempty string")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be a JSON array")
    return value


def _choice(value: Any, path: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {sorted(choices)!r}")
    return value


def _frame_ids(values: Iterable[int], path: str) -> set[int]:
    try:
        values = list(values)
    except TypeError:
        _fail(path, "must be an iterable of nonnegative integer source frame IDs")
    result: set[int] = set()
    for index, frame_id in enumerate(values):
        if type(frame_id) is not int or frame_id < 0:
            _fail(f"{path}[{index}]", "must be a nonnegative integer source frame ID")
        if frame_id in result:
            _fail(f"{path}[{index}]", f"duplicate source frame ID {frame_id}")
        result.add(frame_id)
    return result


def validate_report(report: Any, valid_frame_ids: Iterable[int]) -> dict[str, Any]:
    """Validate the exact evidence-v2 contract against frames actually supplied.

    Frame IDs identify original video frames, not image positions or crop IDs.
    Confirmed physical defects must have evidence spanning at least two distinct
    source frames; two crops of one frame cannot satisfy this requirement.
    Returns the same report without correcting or mutating its content.
    """
    supplied_frames = _frame_ids(valid_frame_ids, "valid_frame_ids")
    report = _object(report, "report", {
        "schema_version", "task", "observations", "task_assessment",
        "physics_assessment", "visual_assessment", "uncertainties",
    })
    if report["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"must equal {SCHEMA_VERSION!r}")
    task = _object(report["task"], "task", {
        "requested_action", "target_description", "final_state_requirement",
    })
    for key, value in task.items():
        _text(value, f"task.{key}")

    observations: dict[str, set[int]] = {}
    for index, raw in enumerate(_list(report["observations"], "observations")):
        path = f"observations[{index}]"
        observation = _object(raw, path, {"id", "frames", "description"})
        evidence_id = _text(observation["id"], f"{path}.id")
        if not re.fullmatch(r"E[1-9][0-9]*", evidence_id):
            _fail(f"{path}.id", "must match E1, E2, ...")
        if evidence_id in observations:
            _fail(f"{path}.id", f"duplicate evidence ID {evidence_id!r}")
        frames = _frame_ids(_list(observation["frames"], f"{path}.frames"), f"{path}.frames")
        if not frames:
            _fail(f"{path}.frames", "must cite at least one supplied source frame")
        if not frames <= supplied_frames:
            _fail(f"{path}.frames", f"unsupplied source frame IDs {sorted(frames - supplied_frames)!r}")
        _text(observation["description"], f"{path}.description")
        observations[evidence_id] = frames

    def evidence(value: Any, path: str, *, required: bool) -> set[int]:
        refs = _list(value, path)
        if required and not refs:
            _fail(path, "observable claims require at least one evidence reference")
        cited: set[str] = set()
        frames: set[int] = set()
        for index, ref in enumerate(refs):
            _text(ref, f"{path}[{index}]")
            if ref not in observations:
                _fail(f"{path}[{index}]", f"unknown evidence ID {ref!r}")
            if ref in cited:
                _fail(f"{path}[{index}]", f"duplicate evidence ID {ref!r}")
            cited.add(ref)
            frames.update(observations[ref])
        return frames

    def assessment(value: Any, path: str, verdicts: set[str], extra: set[str]) -> dict[str, Any]:
        item = _object(value, path, {"verdict", "confidence", "evidence", "reason"} | extra)
        verdict = _choice(item["verdict"], f"{path}.verdict", verdicts | {"unobservable"})
        confidence = _choice(item["confidence"], f"{path}.confidence", {"high", "medium", "low"})
        _text(item["reason"], f"{path}.reason")
        evidence(item["evidence"], f"{path}.evidence", required=verdict != "unobservable")
        if verdict == "unobservable" and confidence != "low":
            _fail(f"{path}.confidence", "unobservable requires low confidence")
        return item

    task_assessment = assessment(report["task_assessment"], "task_assessment", set(TASK_SCORES), {
        "target_match",
    })
    match = _choice(task_assessment["target_match"], "task_assessment.target_match", {
        "match", "mismatch", "uncertain",
    })
    if task_assessment["verdict"] == "complete" and match != "match":
        _fail("task_assessment", "complete requires target_match='match'")
    if task_assessment["verdict"] == "mostly_complete" and match == "mismatch":
        _fail("task_assessment", "mostly_complete is incompatible with target_match='mismatch'")

    physics = assessment(report["physics_assessment"], "physics_assessment", set(PHYSICS_SCORES), {
        "issues",
    })
    confirmed: set[str] = set()
    for index, raw in enumerate(_list(physics["issues"], "physics_assessment.issues")):
        path = f"physics_assessment.issues[{index}]"
        issue = _object(raw, path, {
            "kind", "severity", "certainty", "evidence", "reason", "alternative_explanation",
        })
        _choice(issue["kind"], f"{path}.kind", {"interpenetration", "contact", "shape", "motion", "other"})
        severity = _choice(issue["severity"], f"{path}.severity", {"minor", "major"})
        certainty = _choice(issue["certainty"], f"{path}.certainty", {"confirmed", "uncertain"})
        frames = evidence(issue["evidence"], f"{path}.evidence", required=True)
        _text(issue["reason"], f"{path}.reason")
        _text(issue["alternative_explanation"], f"{path}.alternative_explanation")
        if certainty == "confirmed":
            if len(frames) < 2:
                _fail(f"{path}.evidence", "confirmed physics issue requires two distinct source frames")
            confirmed.add(severity)

    verdict = physics["verdict"]
    if verdict == "major_defect" and "major" not in confirmed:
        _fail("physics_assessment", "major_defect requires a confirmed major issue")
    if verdict == "minor_defect" and ("minor" not in confirmed or "major" in confirmed):
        _fail("physics_assessment", "minor_defect requires confirmed minor issues and no confirmed major issue")
    if verdict in {"plausible", "unobservable"} and confirmed:
        _fail("physics_assessment", f"{verdict} is incompatible with confirmed physics issues")
    assessment(report["visual_assessment"], "visual_assessment", set(VISUAL_SCORES), set())
    for index, uncertainty in enumerate(_list(report["uncertainties"], "uncertainties")):
        _text(uncertainty, f"uncertainties[{index}]")
    return report


def score_report(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a validated report with a provisional monotone rubric.

    Call validate_report with the actual input frame manifest first. This function
    additionally checks internal consistency using the report's own frame IDs, but
    cannot verify which images the caller supplied.

    Reward = (0.70 task + 0.20 physics + 0.10 visual) * (0.5 + 0.5 physics).
    The soft physics factor penalizes confirmed defects without destroying task
    progress. An uncertain issue never becomes a confirmed failure. Unknown
    components produce None, not zero or a fabricated neutral value. Low-confidence
    or materially uncertain judgments carry a review flag even when a provisional
    score exists. The free-text uncertainties list records visibility limitations;
    it is informational, not a separate gate. Material ambiguities must also be
    represented by low confidence, unobservable, uncertain target identity, or an
    uncertain physics issue. No keyword inference is performed on these notes.
    These weights require calibration against independent human labels.
    """
    own_frames: set[int] = set()
    if isinstance(report, dict) and isinstance(report.get("observations"), list):
        for observation in report["observations"]:
            if isinstance(observation, dict) and isinstance(observation.get("frames"), list):
                own_frames.update(frame for frame in observation["frames"] if type(frame) is int)
    validate_report(report, own_frames)
    task = report["task_assessment"]
    physics = report["physics_assessment"]
    visual = report["visual_assessment"]
    components = {
        "task": TASK_SCORES.get(task["verdict"]),
        "physics": PHYSICS_SCORES.get(physics["verdict"]),
        "visual": VISUAL_SCORES.get(visual["verdict"]),
    }
    if task["target_match"] == "mismatch" and components["task"] is not None:
        components["task"] = min(components["task"], 0.2)

    review_reasons: list[str] = []
    for name, item in (("task", task), ("physics", physics), ("visual", visual)):
        if item["verdict"] == "unobservable":
            review_reasons.append(f"{name}: unobservable; no numeric reward can be assigned")
        elif item["confidence"] == "low":
            review_reasons.append(f"{name}: low confidence")
    if task["target_match"] == "uncertain":
        review_reasons.append("task: target identity is uncertain")
    for index, issue in enumerate(physics["issues"]):
        if issue["certainty"] == "uncertain":
            review_reasons.append(f"physics: unresolved {issue['kind']} issue at index {index}")
    total_score = None
    if all(value is not None for value in components.values()):
        task_score, physics_score, visual_score = (components[name] for name in ("task", "physics", "visual"))
        total_score = round(
            (0.70 * task_score + 0.20 * physics_score + 0.10 * visual_score)
            * (0.5 + 0.5 * physics_score), 8,
        )
    return {
        "total_score": total_score,
        "component_scores": components,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "scoring_version": SCORING_VERSION,
    }
