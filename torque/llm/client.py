"""OpenAI-compatible chat for the diagnosis layer.

Backends (LLM_PROVIDER=auto picks the first configured):
  evolus : Evolus agent "Torque Diagnostician" (its model = our vLLM Gemma on AMD) via EVOLUS_CHAT_BASE
  vllm   : our vLLM Gemma endpoint directly (VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL)
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

from torque.config import Settings
from torque.evolus import agent as evolus_agent
from torque.evolus.client import EvolusClient

log = logging.getLogger("torque.llm")
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
    elif mode == "vllm":
        url = f"{s.vllm_base_url}/chat/completions"
        body = {"model": s.vllm_model, "messages": messages, "temperature": 0.2, "max_tokens": 1500}
        headers = {"Authorization": f"Bearer {s.vllm_api_key}"} if s.vllm_api_key else {}
        last = None
        for _ in range(2):  # timeout + one retry
            t1 = time.perf_counter()
            try:
                with httpx.Client(timeout=s.http_timeout_s) as c:
                    r = c.post(url, json=body, headers=headers)
                if store:
                    store.audit("vllm", f"chat.{kind}", "POST", url, r.status_code, (time.perf_counter() - t1) * 1000, "live", r.text[:300])
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                break
            except httpx.HTTPError as e:
                last = e
        else:
            raise RuntimeError(f"vLLM endpoint failed: {last!r}")
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
