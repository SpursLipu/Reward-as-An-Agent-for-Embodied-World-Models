"""Ark Responses adapter preserving the existing reward prompts and sampled images."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time

import httpx

_cooldown_until = 0.0
_dispatch_lock = asyncio.Lock()
_last_dispatch = 0.0

async def paced_dispatch():
    """Spread requests, including retries, instead of bursting after a cooldown."""
    global _last_dispatch
    async with _dispatch_lock:
        while True:
            delay = max(_cooldown_until, _last_dispatch + 1.0) - time.monotonic()
            if delay <= 0:
                _last_dispatch = time.monotonic()
                return
            await asyncio.sleep(delay)



class DoubaoRequestError(RuntimeError):
    """Preserve non-secret request provenance when a provider request fails."""
    def __init__(self, message, metadata):
        super().__init__(message)
        self.sampling_metadata = metadata


def load_ark_key() -> str:
    for name in ("ARK_API_KEY", "REWARD_AS_AGENT_API_KEY"):
        value = os.environ.get(name)
        if value and value != "dummy":
            return value
    source = Path(os.environ.get("ARK_KEY_SOURCE", "/kairos_vepfs_volc/data/zhukangkang/skills/doubao-video/scripts/ark_video.py"))
    if source.is_file():
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets):
                if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                    for value in node.value.values[1:]:
                        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                            return value.value
    raise RuntimeError("No Ark credential available; set ARK_API_KEY or ARK_KEY_SOURCE")


def responses_input(messages):
    """Translate transport fields only; retain roles, text and image order."""
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image_url":
                parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
            else:
                raise ValueError("Unsupported reward content type: " + part["type"])
        result.append({"role": message["role"], "content": parts})
    return result


def build_responses_payload(messages, settings):
    payload = {
        "model": settings.model,
        "input": responses_input(messages),
        "max_output_tokens": settings.max_tokens,
        "thinking": {"type": "disabled"},
    }
    # Older integrations may provide a Settings-like object without the new
    # optional sampling field.  Absence means provider default, exactly like
    # an unset REWARD_AS_AGENT_TEMPERATURE.
    temperature = getattr(settings, "temperature", None)
    if temperature is not None:
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be None or a finite number in [0, 2]')
        payload['temperature'] = temperature
    return payload


def encode_request_payload(payload):
    """These exact UTF-8 JSON bytes are passed as the HTTP request body."""
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sampling_fields(container):
    result = {}
    for field in ('temperature', 'top_p'):
        present = field in container
        value = container.get(field)
        safe = value is None or type(value) in (int, float) and math.isfinite(value)
        result[field] = {'present': present, 'value': value if safe else None}
        if not safe:
            # Do not persist unexpected provider strings in metadata fields.
            result[field]['invalid_value_type'] = type(value).__name__
    return result


def request_metadata(payload, body):
    return {'request_payload_sha256': hashlib.sha256(body).hexdigest(),
            'request_sampling': sampling_fields(payload), 'response_sampling': None,
            'sampling_metadata_status': 'request_recorded_response_unavailable'}


def cached_sampling_metadata(cached, expected):
    required = ('request_payload_sha256', 'request_sampling', 'response_sampling')
    if not all(field in cached for field in required):
        status = 'missing_cached_metadata'
    elif (cached['request_payload_sha256'] != expected['request_payload_sha256']
          or cached['request_sampling'] != expected['request_sampling']
          or not isinstance(cached['response_sampling'], dict)):
        status = 'invalid_cached_metadata'
    else:
        return {field: cached[field] for field in required} | {'sampling_metadata_status': 'recorded'}
    # Do not reconstruct historical request or response evidence from today's configuration.
    return {field: None for field in required} | {'sampling_metadata_status': status}


async def call_doubao(messages, settings):
    global _cooldown_until
    key = load_ark_key()
    url = os.environ.get("ARK_RESPONSES_URL") or settings.api_base.rstrip("/") + "/responses"
    payload = build_responses_payload(messages, settings)
    request_body = encode_request_payload(payload)
    metadata = request_metadata(payload, request_body)
    cache_path = None
    cache_root = None if os.environ.get('REWARD_AS_AGENT_CACHE_BYPASS') == '1' else os.environ.get("REWARD_AS_AGENT_CACHE_DIR")
    if cache_root:
        digest = hashlib.sha256(json.dumps({"url": url, "payload": payload}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache_path = Path(cache_root) / (digest + ".json")
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            cached.update(cached_sampling_metadata(cached, metadata))
            cached["cache_reused"] = True
            return cached
    # Ark needs the machine's outbound proxy, unlike node-local vLLM.
    async with httpx.AsyncClient(trust_env=True) as client:
        for attempt in range(settings.max_retries + 1):
            await paced_dispatch()
            started = time.monotonic()
            try:
                response = await client.post(url, content=request_body,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    timeout=httpx.Timeout(settings.llm_timeout, connect=15))
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt < settings.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                raise DoubaoRequestError(type(exc).__name__, metadata) from None
            except httpx.RequestError as exc:
                raise DoubaoRequestError(type(exc).__name__, metadata) from None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < settings.max_retries:
                    if response.status_code == 429:
                        try:
                            retry_after = float(response.headers.get("Retry-After", "60"))
                        except ValueError:
                            retry_after = 60
                        delay = max(0, retry_after) if math.isfinite(retry_after) else 60
                        _cooldown_until = max(_cooldown_until, time.monotonic() + delay)
                        try:
                            rate_code = str(response.json().get("error", {}).get("code", "unknown")).replace(key, "[REDACTED]")[:100]
                        except Exception:
                            rate_code = "unknown"
                        print(json.dumps({"event": "doubao_rate_limit", "cooldown_seconds": delay, "code": rate_code, "retry_after_provided": "Retry-After" in response.headers}), flush=True)
                    else:
                        await asyncio.sleep(min(2 ** attempt, 30))
                    continue
            if response.is_error:
                # Keep enough API diagnostics for integration errors, never log inputs or keys.
                try:
                    error = response.json().get("error", {})
                    message = str(error.get("message", "")).replace(key, "[REDACTED]")[:500]
                    code = str(error.get("code", "unknown")).replace(key, "[REDACTED]")[:100]
                except Exception:
                    message, code = "non-JSON error", "unknown"
                raise DoubaoRequestError(f"Ark HTTP {response.status_code} {code}: {message}", metadata)
            try:
                body = response.json()
            except ValueError:
                raise DoubaoRequestError('Ark returned non-JSON response', metadata) from None
            if not isinstance(body, dict):
                raise DoubaoRequestError('Ark returned non-object response', metadata)
            metadata = {**metadata, 'response_sampling': sampling_fields(body), 'sampling_metadata_status': 'recorded'}
            if body.get("status") != "completed":
                message = f"Ark response not completed: {body.get('status')}; {body.get('incomplete_details')}"
                raise DoubaoRequestError(message.replace(key, '[REDACTED]')[:600], metadata)
            texts = [p["text"] for item in body.get("output", []) if item.get("type") == "message"
                     for p in item.get("content", []) if p.get("type") == "output_text" and p.get("text")]
            text = "\n".join(texts).strip()
            if not text:
                raise DoubaoRequestError("Ark returned no output text", metadata)
            print(json.dumps({"event": "doubao_call", "response_id": body.get("id"),
                              "seconds": round(time.monotonic() - started, 2),
                              "usage": body.get("usage", {})}), flush=True)
            # Preserve the internal interface expected by retry_llm_call.
            result = {"choices": [{"message": {"content": text}}], "usage": body.get("usage", {}),
                      "id": body.get("id"), "provider_model": body.get("model"), "provider_status": body.get("status"), "cache_reused": False, **metadata}
            if cache_path:
                result["_cache_path"] = str(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp = cache_path.with_suffix("." + str(body.get("id", "tmp")) + ".tmp")
                temp.write_text(json.dumps(result, ensure_ascii=False))
                temp.replace(cache_path)
            return result
    raise RuntimeError("Ark request retries exhausted")
