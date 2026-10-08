"""Score reducers for Reward as An Agent judge outputs."""

from __future__ import annotations

import traceback


def planning_calc_score(result):
    """中文：根据 planning JSON 结果计算分数。
English: Compute the score from the planning JSON result."""
    try:
        return result["video_quality_category"]
    except Exception:
        traceback.print_exc()
        return -1


def physics_reflection_calc_score(result):
    """中文：根据 physics reflection JSON 结果计算分数。
English: Compute the score from the physics reflection JSON result."""
    try:
        s1 = result["interpenetration"]["物体-物体穿模"]
        s2 = result["interpenetration"]["物体-机械臂穿模"]
        s3 = result["shape"]["结构一致性"]
        s4 = result["shape"]["弹性合理性"]
        s5 = result["physics"]["接触一致性"]
        s6 = result["physics"]["动力学一致性"]
        return s1 * s2 * s3 * s4 * s5 * s6
    except Exception:
        traceback.print_exc()
        return -1


def interaction_calc_score(result):
    """中文：根据 interaction JSON 结果计算分数。
English: Compute the score from the interaction JSON result."""
    try:
        s1 = result["接触一致性"]
        s2 = result["动力学一致性"]
        return s1 * s2
    except Exception:
        traceback.print_exc()
        return -1


def shape_calc_score(result):
    """中文：根据 shape JSON 结果计算分数。
English: Compute the score from the shape JSON result."""
    try:
        s1 = result["结构一致性"]
        s2 = result["弹性合理性"]
        return s1 * s2
    except Exception:
        traceback.print_exc()
        return -1


def interpenetration_calc_score(result):
    """中文：根据 interpenetration JSON 结果计算分数。
English: Compute the score from the interpenetration JSON result."""
    try:
        s1 = result["物体-物体穿模"]
        s2 = result["物体-机械臂穿模"]
        return s1 * s2
    except Exception:
        traceback.print_exc()
        return -1


def instruction_following_calc_score(result):
    """中文：根据 instruction following JSON 结果计算分数。
English: Compute the score from the instruction following JSON result."""
    try:
        s1 = (1 - result["视角一致性扣分"]) * 0.5
        s2 = (1 - result["目标一致性扣分"]) * 0.5
        sum_score = sum([s1, s2])
        return round(sum_score, 2)
    except Exception:
        traceback.print_exc()
        return -1


def background_consistency_calc_score(result):
    """中文：根据 background consistency JSON 结果计算分数。
English: Compute the score from the background consistency JSON result."""
    try:
        s1 = result["物品维度得分"] * 0.5
        s2 = result["环境维度得分"] * 0.5
        return round(s1 + s2, 2)
    except Exception:
        traceback.print_exc()
        return -1


def color_calc_score(result):
    """中文：根据 color JSON 结果计算分数。
English: Compute the score from the color JSON result."""
    try:
        return int(result["score"])
    except Exception:
        traceback.print_exc()
        return -1


def clarity_and_brightness_calc_score(result):
    """中文：根据 clarity and brightness JSON 结果计算分数。
English: Compute the score from the clarity and brightness JSON result."""
    try:
        return int(result["清晰度_score"]) * int(result["亮度_score"])
    except Exception:
        traceback.print_exc()
        return -1


def task_calc_score(result):
    """中文：根据 task JSON 结果计算分数。
English: Compute the score from the task JSON result."""
    try:
        return int(result["score"])
    except Exception:
        traceback.print_exc()
        return -1


def task_complete_calc_score(result):
    """中文：根据 task complete JSON 结果计算分数。
English: Compute the score from the task complete JSON result."""
    try:
        # return int(result["process_score"]) * int(result["result_score"])
        return int(result["过程_score"]) * int(result["结果_score"])
    except Exception:
        traceback.print_exc()
        return -1
