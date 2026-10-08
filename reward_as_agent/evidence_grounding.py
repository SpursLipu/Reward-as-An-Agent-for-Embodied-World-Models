"""Mechanical provenance checks; these do not establish visual or human truth."""
from __future__ import annotations

import re

from reward_as_agent.evidence_schema import validate_report


# Observation prose uses source-frame references. Numeric time coordinates are
# rendered from the frame manifest instead of trusting model-authored conversion.
_NUMERIC_TIME = re.compile(
    r'(?<![\d.])\d+(?:\.\d+)?\s*'
    r'(?:秒(?:钟)?|分钟|小时|milliseconds?\b|msecs?\b|ms\b|seconds?\b|secs?\b|s\b|minutes?\b|mins?\b|hours?\b|hrs?\b)',
    re.IGNORECASE,
)


def validate_observation_time_format(report):
    for index, observation in enumerate(report.get('observations', [])):
        description = observation.get('description', '')
        if isinstance(description, str) and _NUMERIC_TIME.search(description):
            raise ValueError(
                f'observations[{index}].description: use original source frame IDs for temporal '
                'locations, not numeric seconds or other numeric time units. The caller derives '
                'timestamps from the supplied frame manifest. Rephrase the observation without '
                'changing visible facts; do not change task duration requirements.'
            )


def core_report(report):
    """Adapt to the unchanged v2 categorical reducer without losing stored checks."""
    return {key: value for key, value in report.items() if key != 'requirement_checks'}


def validate_grounded_report(report, valid_frame_ids, contract):
    from reward_as_agent.task_contract import validate_requirement_checks

    if not isinstance(report, dict) or 'requirement_checks' not in report:
        raise ValueError('The complete report must include requirement_checks for the frozen task contract')
    validate_report(core_report(report), valid_frame_ids)
    if report['task'] != contract['task']:
        raise ValueError('task must exactly equal the frozen task contract task; do not rewrite requirements')
    validate_requirement_checks(report['requirement_checks'], contract, report)
    validate_observation_time_format(report)
    return report


def observation_time_manifest(report, frame_manifest):
    supplied = {frame['source_frame_index']: frame['timestamp_seconds'] for frame in frame_manifest}
    return {
        observation['id']: [
            {'source_frame_index': frame_id, 'timestamp_seconds': supplied[frame_id]}
            for frame_id in observation['frames']
        ]
        for observation in report['observations']
    }
