"""Prompt coverage checks, not evidence of model accuracy."""
import pytest

from reward_as_agent.evidence_prompts import assessment_prompt, verification_prompt, observation_prompt
from reward_as_agent.training_reward import failure_resolution_prompt
from test_evidence_pipeline import contract_fixture

MANIFEST = [{'source_frame_index': 0, 'timestamp_seconds': 0.0, 'view': 'full'},
            {'source_frame_index': 79, 'timestamp_seconds': 3.95, 'view': 'full'}]


@pytest.mark.parametrize('builder', [assessment_prompt, verification_prompt])
def test_all_visual_judgements_distinguish_progress_from_completion(builder):
    prompt = builder(MANIFEST, contract_fixture())
    for text in ('没有有效任务进展', '不可只列失败条件而忽略已有进展',
                 '不能按', 'met 数量/比例机械投票', '进展已完全逆转',
                 '物理/画质缺陷单独评价', '非全部 met 不意味着 failed'):
        assert text in prompt
    assert '0.595' not in prompt
    assert 'demo_' not in prompt


def test_reflection_challenges_both_false_failures_and_unearned_progress():
    prompt = verification_prompt(MANIFEST, contract_fixture())
    assert '对 failed 专门查找反证' in prompt
    assert '对 partial 同样检查' in prompt


def test_failure_resolution_cannot_turn_one_unmet_condition_into_total_failure():
    prompt = failure_resolution_prompt()
    assert '不能单独证明整体 failed' in prompt
    assert '本阶段输出 unresolved' in prompt


def test_blind_observation_stays_task_blind():
    prompt = observation_prompt(MANIFEST)
    assert 'Place block in basket.' not in prompt
    assert '任务类别定义' not in prompt


@pytest.mark.parametrize('builder', [assessment_prompt, verification_prompt])
def test_sampling_limits_alone_are_not_physics_issues(builder):
    prompt = builder(MANIFEST, contract_fixture())
    assert 'confirmed 与 uncertain 均适用' in prompt
    assert '不得据此创建 issue' in prompt
    assert '具体反常时' in prompt
    assert '不能用上述规则自动删除' in prompt
