"""OpenAI-compatible LLM client helpers."""

from __future__ import annotations

import json
import traceback
from contextlib import asynccontextmanager

import httpx

_MODEL_AUXILIARY_PREFIXES = ('reason_', 'explanation_', 'rationale_')


def normalize_model_json(value):
    """Drop non-semantic explanatory extras while preserving strict core fields."""
    removed = []

    def visit(node, path):
        if isinstance(node, dict):
            cleaned = {}
            for key, item in node.items():
                if isinstance(key, str) and key.startswith(_MODEL_AUXILIARY_PREFIXES):
                    removed.append('.'.join(path + [key]))
                    continue
                cleaned[key] = visit(item, path + [str(key)])
            return cleaned
        if isinstance(node, list):
            return [visit(item, path + [str(index)]) for index, item in enumerate(node)]
        return node

    return visit(value, []), removed


class DPRequestRouter:
    """Bound and explicitly balance requests across local vLLM DP engines."""

    def __init__(self, dp_size: int, max_inflight_per_dp: int):
        if dp_size < 1:
            raise ValueError("REWARD_AS_AGENT_DP_SIZE must be at least 1")
        if max_inflight_per_dp < 1:
            raise ValueError("REWARD_AS_AGENT_MAX_INFLIGHT_PER_DP must be at least 1")
        self.dp_size = dp_size
        self._counts = [0] * dp_size
        self._next_rank = 0
        self._lock = None
        self._slots = None
        self._slot_count = dp_size * max_inflight_per_dp

    def _primitives(self):
        # Construct asyncio primitives lazily so importing the app outside a
        # running event loop remains safe.
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()
            self._slots = asyncio.Semaphore(self._slot_count)
        return self._lock, self._slots

    @asynccontextmanager
    async def route(self):
        lock, slots = self._primitives()
        await slots.acquire()
        rank = None
        try:
            async with lock:
                minimum = min(self._counts)
                for offset in range(self.dp_size):
                    candidate = (self._next_rank + offset) % self.dp_size
                    if self._counts[candidate] == minimum:
                        rank = candidate
                        break
                assert rank is not None
                self._counts[rank] += 1
                self._next_rank = (rank + 1) % self.dp_size
            yield rank
        finally:
            if rank is not None:
                async with lock:
                    self._counts[rank] -= 1
            slots.release()


_ROUTERS: dict[tuple[int, int], DPRequestRouter] = {}


def _get_router(settings) -> DPRequestRouter:
    key = (settings.dp_size, settings.max_inflight_per_dp)
    router = _ROUTERS.get(key)
    if router is None:
        router = _ROUTERS[key] = DPRequestRouter(*key)
    return router


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
        try:
            result = json.loads(content)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # Some models append commentary or even a second JSON object. Decode
        # the first complete object instead of using a greedy ``{.*}`` match,
        # which turns otherwise usable output into "Extra data" failures.
        object_start = content.find("{")
        if object_start < 0:
            return None
        result, _ = json.JSONDecoder().raw_decode(content[object_start:])
        return result if isinstance(result, dict) else None
    except Exception:
        traceback.print_exc()
        return None


async def call_llm(messages, settings):
    """中文：调用 OpenAI 兼容的 chat completions 接口。
English: Call an OpenAI-compatible chat completions endpoint."""
    if settings.provider == "doubao":
        from reward_as_agent.doubao import call_doubao
        async with _get_router(settings).route():
            return await call_doubao(messages, settings)
    if settings.provider != "openai":
        raise ValueError(f"Unknown reward provider: {settings.provider}")
    router = _get_router(settings)
    async with router.route() as dp_rank:
        payload = {
            "model": settings.model,
            "messages": messages,
            "max_tokens": settings.max_tokens,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        if settings.dp_size > 1:
            # vLLM 0.19 accepts this header and bypasses its imbalanced
            # internal DP load balancer for this request.
            headers["X-data-parallel-rank"] = str(dp_rank)
        # Reward-node traffic is always node-local. Ignore cluster-wide proxy
        # variables so a missing NO_PROXY entry cannot send vLLM requests via
        # Squid (or fail with a proxy-side timeout).
        async with httpx.AsyncClient(trust_env=False) as client:
            from reward_as_agent.doubao import encode_request_payload, request_metadata, sampling_fields
            request_body = encode_request_payload(payload)
            resp = await client.post(
                f"{settings.api_base}/chat/completions",
                content=request_body,
                headers={**headers, 'Content-Type': 'application/json'},
                timeout=settings.llm_timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            result.update(request_metadata(payload, request_body))
            result['response_sampling'] = sampling_fields(result)
            result['sampling_metadata_status'] = 'recorded'
            return result


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
        if settings.provider == "doubao" and (json_output is None or score == -1) and res.get("_cache_path"):
            from pathlib import Path
            Path(res["_cache_path"]).unlink(missing_ok=True)
        retried += 1

    if settings.provider == "doubao" and (json_output is None or score == -1):
        raise RuntimeError("Doubao returned invalid scoring JSON after bounded retries")
    if json_output is not None:
        api_output = {"index": idx, "score": score, "status": "success"}
        if settings.provider == "doubao":
            api_output["usage"] = res.get("usage", {})
            api_output["response_id"] = res.get("id")
            api_output["cache_reused"] = res.get("cache_reused", False)
    else:
        print("达到最大次数仍然失败")
        print("模型最后一次返回结果：")
        print(origin_output)
        api_output = {"index": idx, "score": -1, "status": "max_retry_failed"}
    return json_output, api_output
