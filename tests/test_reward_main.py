import pytest
from fastapi.testclient import TestClient

from reward_as_agent import app as reward_app
from reward_as_agent import pipeline


def test_sample_uniform_frames_handles_empty_input():
    """中文：验证空帧序列采样会返回空列表。
English: Verify that sampling an empty frame sequence returns an empty list."""
    assert pipeline.sample_uniform_frames([], 4) == []


def test_sample_uniform_frames_keeps_first_and_last():
    """中文：验证均匀采样会保留首帧和尾帧。
English: Verify that uniform sampling preserves the first and last frames."""
    frames = ["f0", "f1", "f2", "f3", "f4"]

    sampled = pipeline.sample_uniform_frames(frames, 3)

    assert sampled[0] == "f0"
    assert sampled[-1] == "f4"
    assert len(sampled) == 3


def test_health_endpoint():
    """中文：验证健康检查接口返回服务状态。
English: Verify that the health endpoint returns service status."""
    client = TestClient(reward_app.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "api_base" in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"video_path": [], "prompt": "task"},
        {"video_path": ["/tmp/not-exist.mp4"], "prompt": "task"},
        {"video_path": ["/tmp/not-exist.mp4"], "prompt": ""},
    ],
)
def test_eval_video_rejects_bad_requests(payload):
    """中文：验证评估接口会拒绝非法请求。
English: Verify that the evaluation endpoint rejects invalid requests."""
    client = TestClient(reward_app.app)

    response = client.post("/eval_video", json=payload)

    assert response.status_code == 400


def test_planning_weight_gate_values():
    """中文：验证 planning 门控权重与当前评分策略一致。
English: Verify planning gate weights match the current scoring policy."""
    assert pipeline.planning_weight("Very Poor") is None
    assert pipeline.planning_weight("Moderate Issues") == 0.6
    assert pipeline.planning_weight("Excellent") == 1.0


def test_weighted_score_uses_stage_weights():
    """中文：验证总分按各阶段权重聚合。
English: Verify final scoring aggregates stage weights correctly."""
    score = pipeline.weighted_score(
        planning_gate=0.6,
        vision_score=1,
        physics_score=1,
        instruction_score=0.7,
        task_score=0,
    )

    assert score == pytest.approx(0.384)


def test_response_from_result_maps_success_and_error():
    """中文：验证内部结果会转换为稳定的流式 API 输出。
English: Verify internal results convert to stable streamed API outputs."""
    success = {
        "planning_api_output": {"index": 2, "score": "Excellent", "status": "success"},
        "total_score": 0.7,
    }
    error = {
        "planning_api_output": {"index": 3, "score": -1, "status": "error"},
        "total_score": -1,
        "error": "boom",
    }

    assert reward_app.response_from_result(success) == {
        "index": 2,
        "score": 0.7,
        "status": "success",
    }
    assert reward_app.response_from_result(error) == {
        "index": 3,
        "score": -1,
        "status": "error",
        "error": "boom",
    }
