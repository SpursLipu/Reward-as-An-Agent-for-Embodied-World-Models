"""FastAPI application for Reward as An Agent."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reward_as_agent.config import get_settings
from reward_as_agent.logging import save_result_to_txt_table
from reward_as_agent.logging import save_video_and_prompt as save_video_and_prompt_base
from reward_as_agent.metrics import MotionQualityMetrics
from reward_as_agent.evidence_pipeline import EvidencePipeline
from reward_as_agent.pipeline import process_one_video_safe


SETTINGS = get_settings()
MOTION_QUALITY = MotionQualityMetrics(SETTINGS)
MOTION_QUALITY_AVAILABLE = MOTION_QUALITY.available
PIPELINE_NAME = 'evidence'
PIPELINE = None


@asynccontextmanager
async def lifespan(app):
    global PIPELINE
    from reward_as_agent.tool_runtime import configured_hook
    hook, client = configured_hook(SETTINGS)
    try:
        ready = await client.evaluate({'operation': 'health'})
        if ready.get('status') != 'ok':
            raise RuntimeError('WMReward startup failed; inspect wmreward.stderr.log')
        PIPELINE = EvidencePipeline(SETTINGS, physics_hook=hook)
        yield
    finally:
        PIPELINE = None
        await client.close()

app = FastAPI(
    title="Reward as An Agent for Embodied World Models",
    description="Agentic reward evaluation service for embodied world-model videos.",
    version="0.1.0",
    lifespan=lifespan,
)


class VideoRequest(BaseModel):
    """中文：/eval_video 接口的请求体结构。
English: Request body schema for the /eval_video endpoint."""

    video_path: list[str]
    prompt: str
    return_details: bool = False


async def save_video_and_prompt(video_list: list[str], shared_prompt: str):
    """中文：使用全局日志目录保存本次请求的视频与 prompt。
English: Save request videos and prompt under the global log directory."""
    return await save_video_and_prompt_base(video_list, shared_prompt, str(SETTINGS.log_root))


def validate_request(data: VideoRequest) -> None:
    """中文：校验请求中的视频路径和任务 prompt 是否有效。
English: Validate video paths and task prompt in the incoming request."""
    if not data.video_path:
        raise HTTPException(status_code=400, detail="video_path must contain at least one video.")
    if not data.prompt or not data.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be a non-empty string.")
    missing = [path for path in data.video_path if not Path(path).is_file()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "Some video paths do not exist.", "missing": missing},
        )


def response_from_result(result: dict) -> dict:
    """中文：将内部完整结果转换为流式 API 返回对象。
English: Convert a full internal result into one streamed API response object."""
    total_score = result["total_score"]
    planning_api_output = result["planning_api_output"]
    # Add the training contract only for structured results; legacy error and
    # compatibility responses keep their existing shape.
    extra = {}
    if 'diagnostic_score' in result:
        extra = {
            'task_verdict': result['evidence_report']['task_assessment']['verdict'],
            'training_eligible': result['training_eligible'],
            'diagnostic_score': result['diagnostic_score'],
            'diagnostic_review_required': result['diagnostic_review_required'],
            'diagnostic_review_reasons': result['diagnostic_review_reasons'],
        }
    if result.get('review_required'):
        return {'index': planning_api_output['index'], 'score': None,
                'provisional_score': total_score if total_score != -1 else None,
                'status': 'needs_review', 'review_reasons': result.get('scoring', {}).get('review_reasons', []),
                **extra}
    if total_score != -1:
        return {
            "index": planning_api_output["index"],
            "score": total_score,
            "status": "success",
            **extra,
        }

    response = {
        "index": planning_api_output["index"],
        "score": -1,
        "status": "error" if result.get("error") else "max_retry_failed",
    }
    if result.get("error"):
        response["error"] = result["error"]
    return response


async def iter_evaluation_results(
    video_paths: list[str],
    description: str,
    current_folder: str | None,
    return_details: bool = False,
):
    """中文：按完成顺序异步产出每个视频的 JSONL 结果。
English: Yield each video's JSONL result asynchronously as it completes."""
    pending = {
        asyncio.create_task(process_one_video_safe(PIPELINE, video_path, description, idx))
        for idx, video_path in enumerate(video_paths)
    }
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                timeout=SETTINGS.heartbeat_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                # The training client ignores records without a video index,
                # while the bytes keep its streamed read timeout alive.
                yield json.dumps({"status": "processing"}) + "\n"
                continue
            for task in done:
                result = task.result()
                if current_folder is not None:
                    if result.get('pipeline_version'):
                        path = Path(current_folder) / ('evidence_' + str(result['planning_api_output']['index']) + '.json')
                        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    elif "error" not in result:
                        save_result_to_txt_table(current_folder, result)
                response = response_from_result(result)
                if return_details:
                    response["details"] = result
                    response["model"] = SETTINGS.model
                    response["provider"] = SETTINGS.provider
                yield json.dumps(response, ensure_ascii=False) + "\n"
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@app.get("/health")
async def health():
    """中文：返回服务健康状态和关键运行配置。
English: Return service health status and key runtime configuration."""
    return {
        "status": "ok" if PIPELINE is not None else "not_ready",
        "model": SETTINGS.model,
        "provider": SETTINGS.provider,
        "pipeline": PIPELINE_NAME,
        "external_tools": ["wmreward"],
        "tool_reflection_required": True,
        "tool_runtime_initialized": PIPELINE is not None,
        "api_base": SETTINGS.api_base,
        "dp_size": SETTINGS.dp_size,
        "max_inflight_per_dp": SETTINGS.max_inflight_per_dp,
        "max_tokens": SETTINGS.max_tokens,
        "heartbeat_interval": SETTINGS.heartbeat_interval,
        "motion_quality_enabled": SETTINGS.enable_motion_quality,
        "motion_quality_available": MOTION_QUALITY_AVAILABLE,
        "log_root": str(SETTINGS.log_root),
    }


@app.post("/eval_video")
async def eval_video(data: VideoRequest):
    """中文：接收批量视频评估请求，并以 JSONL 流式返回 reward 分数。
English: Accept a batch video evaluation request and stream reward scores as JSONL."""
    validate_request(data)
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Required WMReward runtime is not initialized")
    current_folder = None
    if SETTINGS.save_inputs:
        current_folder = await save_video_and_prompt(data.video_path, data.prompt)

    return StreamingResponse(
        iter_evaluation_results(data.video_path, data.prompt, current_folder, data.return_details),
        media_type="application/jsonl; charset=utf-8",
    )
