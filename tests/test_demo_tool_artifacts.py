"""Published demos must contain real matching-source tool/Reflection records."""
import json

import pytest

from scripts.run_demos import DEMOS, REPO_ROOT, load_case, sha256_file
from scripts.run_wmreward_demo import inspect_tool_completion


@pytest.mark.parametrize('demo', DEMOS)
def test_published_demo_has_completed_tool_reflection(demo):
    root = REPO_ROOT / 'examples' / demo
    response = json.loads((root / 'response.json').read_text())
    metadata = json.loads((root / 'run.json').read_text())
    tools = json.loads((root / 'tools.json').read_text())
    reports = json.loads((root / 'reports.json').read_text())
    _, inputs = load_case(demo)
    details = response['details']
    evidence, errors = inspect_tool_completion(details, details['trace'], sha256_file(inputs[-1]))
    assert not errors
    assert evidence == tools
    assert metadata['execution_error'] is None
    assert metadata['inputs_unchanged']
    assert metadata['agent_sources_unchanged']
    assert metadata['configuration']['tool_reflection_required']
    assert reports['before_tool_reflection'] == evidence['pre_tool_report']
    assert reports['after_tool_reflection']
    assert reports['final_after_scope_audit'] == details['evidence_report']
