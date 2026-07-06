"""Reward pipeline orchestration for Reward as An Agent."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable

from reward_as_agent.config import Settings
from reward_as_agent.llm import retry_llm_call as retry_llm_call_base
from reward_as_agent.video import (
    build_content,
    build_physics_reflection_content,
    extract_all_frames,
    sample_uniform_frames,
)
from reward_as_agent.prompts import (
    create_background_consistency_prompt,
    create_clarity_and_brightness_prompt,
    create_instruction_following_prompt,
    create_instruction_following_reflection_prompt,
    create_interaction_prompt,
    create_interpenetration_prompt,
    create_physics_reflection_prompt,
    create_planning_prompt,
    create_shape_prompt,
    create_task_complete_prompt,
)
from reward_as_agent.scoring import (
    background_consistency_calc_score,
    clarity_and_brightness_calc_score,
    instruction_following_calc_score,
    interaction_calc_score,
    interpenetration_calc_score,
    physics_reflection_calc_score,
    planning_calc_score,
    shape_calc_score,
    task_complete_calc_score,
)


PLANNING_FRAMES = 20
VISION_FRAMES = 10
PHYSICS_FRAMES = 64
INSTRUCTION_FRAMES = 20

PLANNING_MODERATE_WEIGHT = 0.6
VISION_WEIGHT = 0.2
PHYSICS_WEIGHT = 0.3
INSTRUCTION_WEIGHT = 0.2
TASK_WEIGHT = 0.3

RESULT_KEYS = (
    "planning_json_output",
    "planning_api_output",
    "clarity_and_brightness_json_output",
    "clarity_and_brightness_api_output",
    "clarity_and_brightness_reflection_json_output",
    "clarity_and_brightness_reflection_api_output",
    "color_json_output",
    "color_api_output",
    "first_frame_json_output",
    "first_frame_api_output",
    "abnormal_frame_json_output",
    "abnormal_frame_api_output",
    "instruction_following_json_output",
    "instruction_following_api_output",
    "instruction_reflection_following_json_output",
    "instruction_reflection_following_api_output",
    "interaction_json_output",
    "interaction_output",
    "interpenetration_json_output",
    "interpenetration_output",
    "shape_json_output",
    "shape_output",
    "physics_reflection_json_out",
    "physics_reflection_out",
    "task_json_output",
    "task_api_output",
    "background_consistency_json_output",
    "background_consistency_api_output",
    "motion_quality_score",
)


class RewardPipeline:
    """中文：封装单视频 reward 评估流程。
English: Encapsulate the single-video reward evaluation pipeline."""

    def __init__(self, settings: Settings):
        """中文：保存运行时配置，用于后续 LLM 调用。
English: Store runtime settings for later LLM calls."""
        self.settings = settings

    async def retry_llm_call(self, messages: list[dict], idx: int, calc_score: Callable):
        """中文：使用当前 pipeline 配置执行带重试的模型调用与计分。
English: Run a retried model call and scoring step with this pipeline's settings."""
        return await retry_llm_call_base(messages, idx, calc_score, self.settings)

    async def run_simple_judge(
        self,
        system_prompt: str,
        frames: list[str],
        idx: int,
        calc_score: Callable,
        description: str | None = None,
    ):
        """中文：执行常规 prompt + frames 的 LLM reward 子模块。
English: Run a regular prompt-and-frames LLM reward submodule."""
        return await self.retry_llm_call(
            simple_messages(system_prompt, frames, description),
            idx,
            calc_score,
        )

    async def planning(self, description: str, frames: list[str], idx: int):
        """中文：评估视频是否具备继续执行完整 reward 流程的基础质量。
English: Judge whether the video has enough baseline quality for the full reward pipeline."""
        return await self.run_simple_judge(
            create_planning_prompt(),
            frames,
            idx,
            planning_calc_score,
            description,
        )

    async def clarity_and_brightness(self, frames: list[str], idx: int):
        """中文：评估视频帧的清晰度与亮度质量。
English: Evaluate frame clarity and brightness quality."""
        return await self.run_simple_judge(
            create_clarity_and_brightness_prompt(),
            frames,
            idx,
            clarity_and_brightness_calc_score,
        )

    async def vision_quality(self, frames: list[str], idx: int) -> dict:
        """中文：聚合视觉质量相关 reward 子模块的评估结果。
English: Aggregate reward submodules related to visual quality."""
        clarity_json, clarity_api = await self.clarity_and_brightness(frames, idx)
        return {
            "clarity_and_brightness_json_output": clarity_json,
            "clarity_and_brightness_api_output": clarity_api,
            "clarity_and_brightness_reflection_json_output": None,
            "clarity_and_brightness_reflection_api_output": None,
            "color_json_output": None,
            "color_api_output": None,
            "first_frame_json_output": None,
            "first_frame_api_output": None,
            "abnormal_frame_json_output": None,
            "abnormal_frame_api_output": None,
        }

    async def interaction(self, frames: list[str], idx: int):
        """中文：评估物体交互和接触动态是否合理。
English: Evaluate whether object interactions and contact dynamics are plausible."""
        return await self.run_simple_judge(create_interaction_prompt(), frames, idx, interaction_calc_score)

    async def interpenetration(self, frames: list[str], idx: int):
        """中文：评估物体之间以及物体与机械臂之间的穿模问题。
English: Evaluate interpenetration between objects and between objects and robot arms."""
        return await self.run_simple_judge(
            create_interpenetration_prompt(),
            frames,
            idx,
            interpenetration_calc_score,
        )

    async def shape(self, frames: list[str], idx: int):
        """中文：评估物体形状结构与弹性变化是否合理。
English: Evaluate whether object structure and elastic deformation are plausible."""
        return await self.run_simple_judge(create_shape_prompt(), frames, idx, shape_calc_score)

    async def physics_reflection(
        self,
        frames: list[str],
        idx: int,
        interaction_output,
        interpenetration_output,
        shape_output,
    ):
        """中文：整合交互、穿模与形变判断，生成物理一致性的反思评分。
English: Combine interaction, interpenetration, and shape judgments into a reflected physics score."""
        messages = [
            {"role": "system", "content": create_physics_reflection_prompt()},
            {
                "role": "user",
                "content": build_physics_reflection_content(
                    frames,
                    interaction_output,
                    interpenetration_output,
                    shape_output,
                ),
            },
        ]
        return await self.retry_llm_call(messages, idx, physics_reflection_calc_score)

    async def physics(self, frames: list[str], idx: int) -> dict:
        """中文：聚合交互、穿模、形变与反思模块，计算物理/运动一致性。
English: Aggregate interaction, interpenetration, shape, and reflection modules for physics/motion consistency."""
        interaction_json, interaction_api = await self.interaction(frames, idx)
        interpenetration_json, interpenetration_api = await self.interpenetration(frames, idx)
        shape_json, shape_api = await self.shape(frames, idx)
        reflection_json, reflection_api = await self.physics_reflection(
            frames,
            idx,
            interaction_json,
            interpenetration_json,
            shape_json,
        )
        return {
            "interaction_json_output": interaction_json,
            "interaction_output": interaction_api,
            "interpenetration_json_output": interpenetration_json,
            "interpenetration_output": interpenetration_api,
            "shape_json_output": shape_json,
            "shape_output": shape_api,
            "physics_reflection_json_out": reflection_json,
            "physics_reflection_out": reflection_api,
        }

    async def instruction_following(self, description: str, frames: list[str], idx: int):
        """中文：评估视频内容对用户任务指令的跟随程度。
English: Evaluate how well the video follows the user task instruction."""
        return await self.run_simple_judge(
            create_instruction_following_prompt(),
            frames,
            idx,
            instruction_following_calc_score,
            description,
        )

    async def instruction_following_reflection(self, initial_result, frames: list[str], idx: int):
        """中文：基于首次指令跟随结果进行反思校正。
English: Refine the initial instruction-following judgment through reflection."""
        initial_result_text = json.dumps(initial_result, ensure_ascii=False)
        messages = [
            {"role": "system", "content": create_instruction_following_prompt()},
            {"role": "user", "content": build_content(frames, initial_result_text)},
            {"role": "assistant", "content": initial_result_text},
            {"role": "user", "content": create_instruction_following_reflection_prompt()},
        ]
        return await self.retry_llm_call(messages, idx, instruction_following_calc_score)

    async def background_consistency(self, description: str, frames: list[str], idx: int):
        """中文：评估物体与环境背景是否与任务描述保持一致。
English: Evaluate whether objects and background remain consistent with the task description."""
        return await self.run_simple_judge(
            create_background_consistency_prompt(),
            frames,
            idx,
            background_consistency_calc_score,
            description,
        )

    async def task_complete(self, description: str, frames: list[str], idx: int):
        """中文：综合过程与结果维度评估任务完成度。
English: Evaluate task completion from both process and outcome dimensions."""
        return await self.run_simple_judge(
            create_task_complete_prompt(),
            frames,
            idx,
            task_complete_calc_score,
            description,
        )

    async def process_one_video(self, video_path: str, description: str, idx: int) -> dict:
        """中文：对单个视频运行完整 Reward as An Agent 流程并返回结构化结果。
English: Run the full Reward as An Agent pipeline for one video and return structured results."""
        _, openai_frames = extract_all_frames(video_path)
        print("总帧数：", len(openai_frames))

        result = init_result()

        planning_frames = sample_uniform_frames(openai_frames, PLANNING_FRAMES)
        result["planning_json_output"], result["planning_api_output"] = await self.planning(
            description,
            planning_frames,
            idx,
        )

        gate = planning_weight(result["planning_api_output"]["score"])
        if gate is None:
            result["total_score"] = 0
            return result

        vision_frames = sample_uniform_frames(openai_frames, VISION_FRAMES)
        result.update(await self.vision_quality(vision_frames, idx))

        if result["clarity_and_brightness_api_output"]["score"] == -1:
            result["total_score"] = -1
            return result
        vision_score = result["clarity_and_brightness_api_output"]["score"]

        # 当前开源版本暂不启用额外动作质量指标；后续接入真实指标后再纳入总分。
        # Motion-quality metrics are disabled in the current release until a real comparison metric is integrated.

        physics_frames = sample_uniform_frames(openai_frames, PHYSICS_FRAMES)
        result.update(await self.physics(physics_frames, idx))

        if result["physics_reflection_out"]["score"] == -1:
            result["total_score"] = weighted_score(gate, vision_score)
            return result

        physics_score = result["physics_reflection_out"]["score"]
        if physics_score < 1:
            result["total_score"] = weighted_score(gate, vision_score, physics_score)
            return result

        instruction_frames = sample_uniform_frames(openai_frames, INSTRUCTION_FRAMES)
        result["instruction_following_json_output"], result["instruction_following_api_output"] = (
            await self.instruction_following(description, instruction_frames, idx)
        )
        (
            result["instruction_reflection_following_json_output"],
            result["instruction_reflection_following_api_output"],
        ) = await self.instruction_following_reflection(
            result["instruction_following_json_output"],
            instruction_frames,
            idx,
        )
        result["background_consistency_json_output"], result["background_consistency_api_output"] = (
            await self.background_consistency(description, instruction_frames, idx)
        )

        if (
            result["instruction_reflection_following_api_output"]["score"] == -1
            or result["background_consistency_api_output"]["score"] == -1
        ):
            result["total_score"] = weighted_score(gate, vision_score, physics_score)
            return result

        instruction_score = (
            0.4 * result["instruction_reflection_following_api_output"]["score"]
            + 0.6 * result["background_consistency_api_output"]["score"]
        )

        result["task_json_output"], result["task_api_output"] = await self.task_complete(
            description,
            instruction_frames,
            idx,
        )
        if result["task_api_output"]["score"] == -1:
            result["total_score"] = weighted_score(gate, vision_score, physics_score, instruction_score)
            return result

        task_score = result["task_api_output"]["score"]
        result["total_score"] = weighted_score(
            gate,
            vision_score,
            physics_score,
            instruction_score,
            task_score,
        )
        return result


def init_result() -> dict:
    """中文：创建包含所有 reward 模块字段的结果字典。
English: Create a result dictionary containing all reward module fields."""
    result = {key: None for key in RESULT_KEYS}
    result["total_score"] = 0
    return result


def simple_messages(system_prompt: str, frames: list[str], description: str | None = None) -> list[dict]:
    """中文：构造常规单轮多模态评估消息。
English: Build a standard single-turn multimodal judge message."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_content(frames, description)},
    ]


def planning_weight(planning_score) -> float | None:
    """中文：将 planning 结果转换为总分门控权重。
English: Convert a planning result into the final-score gate weight."""
    if planning_score == "Very Poor":
        return None
    if planning_score == "Moderate Issues":
        return PLANNING_MODERATE_WEIGHT
    return 1.0


def weighted_score(
    planning_gate: float,
    vision_score: float,
    physics_score: float = 0,
    instruction_score: float = 0,
    task_score: float = 0,
) -> float:
    """中文：按当前模块权重计算已完成阶段的 reward 总分。
English: Compute the reward score for the stages that have completed."""
    return (
        vision_score * VISION_WEIGHT
        + physics_score * PHYSICS_WEIGHT
        + instruction_score * INSTRUCTION_WEIGHT
        + task_score * TASK_WEIGHT
    ) * planning_gate


async def process_one_video_safe(
    pipeline: RewardPipeline,
    video_path: str,
    description: str,
    idx: int,
):
    """中文：带异常保护地评估单个视频，失败时返回标准错误结果。
English: Evaluate one video with exception protection and return a standard error result on failure."""
    try:
        return await pipeline.process_one_video(video_path, description, idx)
    except Exception as exc:
        traceback.print_exc()
        return {
            "planning_api_output": {"index": idx, "score": -1, "status": "error"},
            "total_score": -1,
            "error": str(exc),
        }
