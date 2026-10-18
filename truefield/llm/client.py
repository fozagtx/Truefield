"""OpenAI-compatible chat for the diagnosis layer.

Backends (LLM_PROVIDER=auto picks the first configured):
  do     : PRIMARY. OpenAI-compatible inference API (MODEL_BASE_URL, default https://inference.do-ai.run/v1;
           MODEL_ACCESS_KEY; MODEL_NAME, default gpt-4o). System prompt is sent with role "developer".
  evolus : optional. Evolus agent "Truefield Diagnostician" via EVOLUS_CHAT_BASE
  vllm   : optional. Our own vLLM endpoint (VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL), e.g. Gemma on AMD
  mock   : deterministic local responder so the demo runs end to end offline
DEMO_CACHE=1 replays recorded responses from DATA_DIR/demo_cache (labelled "cached").
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

from truefield.config import Settings
from truefield.evolus import agent as evolus_agent
from truefield.evolus.client import EvolusClient

log = logging.getLogger("truefield.llm")
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def extract_json(text: str):
    """Parse the first JSON object in a model reply (tolerates code fences / stray prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    depth = 0
    for i in range(start, len(t)) if start >= 0 else []:
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("no JSON object found in model output")


class ChatResult(dict):
    pass


# Endpoints that rejected response_format once are remembered so we don't keep paying a failed call.
_NO_RESPONSE_FORMAT: set[str] = set()


def openai_chat(s: Settings, url: str, key: str, model: str, messages: list[dict], kind: str, store, service: str,
                json_mode: bool = False) -> str:
    """POST an OpenAI-compatible chat completion; timeout + one retry. If json_mode and the endpoint rejects
    response_format (HTTP 400/422 mentioning it), resend once without it. Never logs the key."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    use_rf = json_mode and url not in _NO_RESPONSE_FORMAT
    last: Exception | None = None
    attempts = 0
    while attempts < 2:
        body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1500}
        if use_rf:
            body["response_format"] = {"type": "json_object"}
        t1 = time.perf_counter()
        try:
            with httpx.Client(timeout=s.http_timeout_s) as c:
                r = c.post(url, json=body, headers=headers)
            if store:
                store.audit(service, f"chat.{kind}", "POST", url, r.status_code, (time.perf_counter() - t1) * 1000, "live", r.text[:300])
            if use_rf and r.status_code in (400, 422) and "response_format" in r.text.lower():
                log.info("%s rejected response_format; retrying without it", url)
                _NO_RESPONSE_FORMAT.add(url)
                use_rf = False
                continue  # does not count as a retry
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
                attempts += 1
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"{service} endpoint error: HTTP {e.response.status_code}") from None
        except httpx.HTTPError as e:
            last = e
            attempts += 1
    raise RuntimeError(f"{service} endpoint failed: {last!r}")


def _cache_path(s: Settings, kind: str) -> Path:
    return s.data_dir / "demo_cache" / f"{kind}.json"


def chat(s: Settings, ec: EvolusClient, messages: list[dict], kind: str, mock_fn, store=None) -> ChatResult:
    """Returns ChatResult(content, provider, cached, latency_ms, conversation_id)."""
    t0 = time.perf_counter()
    if s.demo_cache and _cache_path(s, kind).exists():
        content = json.loads(_cache_path(s, kind).read_text())["content"]
        if store:
            store.audit("llm", f"chat.{kind}", "CACHE", str(_cache_path(s, kind)), 200, 0, "cached", "")
        return ChatResult(content=content, provider="cached", cached=True, latency_ms=0.0, conversation_id=None)
    mode = s.llm_mode
    conv = None
    if mode == "evolus":
        content, conv = evolus_agent.chat(ec, messages)
    elif mode == "do":
        msgs = [dict(m, role="developer") if m["role"] == "system" else m for m in messages]
        content = openai_chat(s, f"{s.model_base_url}/chat/completions", s.model_access_key, s.model_name, msgs, kind, store, "do", json_mode=True)
    elif mode == "vllm":
        content = openai_chat(s, f"{s.vllm_base_url}/chat/completions", s.vllm_api_key, s.vllm_model, messages, kind, store, "vllm")
    else:
        content = mock_fn(messages)
        if store:
            store.audit("llm", f"chat.{kind}", "MOCK", "mock://diagnostician", 200, 0, "mock", "")
    ms = (time.perf_counter() - t0) * 1000
    if s.demo_cache_record:
        p = _cache_path(s, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"content": content, "provider": mode}, indent=2))
    return ChatResult(content=content, provider=mode, cached=False, latency_ms=ms, conversation_id=conv)
