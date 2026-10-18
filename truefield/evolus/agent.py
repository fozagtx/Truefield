"""Evolus agent through the OpenAI-compatible endpoint (verified): POST {EVOLUS_CHAT_BASE}/chat/completions
with model "<applicationId>:<agentId>"; keep X-Conversation-Id for follow-ups. Temperature is set in the
agent settings in the workspace (request params like temperature are ignored)."""
from __future__ import annotations

from truefield.evolus.client import EvolusClient


def list_models(c: EvolusClient) -> list:
    if c.mock:
        c.audit_mock("agent.list_models", "/models")
        return [{"id": "mock-app:mock-agent"}]
    r = c.request("agent.list_models", "GET", "/models", base="chat")
    return c.json_or_raise(r, "agent.list_models").get("data", [])


def chat(c: EvolusClient, messages: list[dict], conversation_id: str | None = None) -> tuple[str, str | None]:
    """Returns (assistant content, conversation id). Caller handles mock mode before calling."""
    headers = {"X-Conversation-Id": conversation_id} if conversation_id else {}
    body = {"model": c.s.evolus_agent_model, "messages": messages, "stream": False}
    r = c.request("agent.chat", "POST", "/chat/completions", base="chat", json=body, headers=headers)
    data = c.json_or_raise(r, "agent.chat")
    content = data["choices"][0]["message"]["content"]
    return content, r.headers.get("X-Conversation-Id") or data.get("conversation_id")
