"""OpenAI-compatible LLM client helpers."""

from __future__ import annotations

import json
import re
import traceback

import httpx


def clean_output(text):
    """中文：去除模型输出中的 Markdown 代码块包裹。
English: Strip Markdown code-fence wrappers from a model response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_parse_json(content):
    """中文：从模型文本中提取并解析 JSON，失败时返回 None。
English: Extract and parse a JSON object from model text, returning None on failure."""
    try:
        content = clean_output(content.strip())
        json_pattern = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_pattern:
            return None
        return json.loads(json_pattern.group(0))
    except Exception:
        traceback.print_exc()
        return None


async def call_llm(messages, settings):
    """中文：调用 OpenAI 兼容的 chat completions 接口。
English: Call an OpenAI-compatible chat completions endpoint."""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": settings.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        resp = await client.post(
            f"{settings.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout,
        )
        return resp.json()


async def retry_llm_call(messages, idx, calc_score, settings):
    """中文：重试模型调用直到得到可解析且可计分的 JSON 结果。
English: Retry model calls until the response is parseable and scoreable."""
    origin_output = None
    json_output = None
    retried = 0
    score = -1

    while json_output is None or score == -1:
        if retried > settings.max_retries:
            break

        if retried != 0:
            print(f"发生异常，正在尝试第{retried}次")

        res = await call_llm(messages, settings)
        if "choices" in res and len(res["choices"]) > 0:
            message = res["choices"][0].get("message", {})
            origin_output = message.get("content", "")
        else:
            print("【规划模块错误】模型返回异常：", res)
            origin_output = '{"video_quality_category": "Error", "reason": "API response error"}'

        json_output = safe_parse_json(origin_output)
        score = calc_score(json_output)
        retried += 1

    if json_output is not None:
        api_output = {"index": idx, "score": score, "status": "success"}
    else:
        print("达到最大次数仍然失败")
        print("模型最后一次返回结果：")
        print(origin_output)
        api_output = {"index": idx, "score": -1, "status": "max_retry_failed"}
    return json_output, api_output
