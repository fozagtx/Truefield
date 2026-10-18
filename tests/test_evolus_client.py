"""Live-mode request shapes against a fake Evolus (httpx.MockTransport). No network."""
import json

import httpx
import pytest

from torque.config import get_settings
from torque.evolus import agent, extractor, library, tools, workflow
from torque.evolus import client as client_mod
from torque.evolus.client import EvolusClient, EvolusError


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("EVOLUS_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUS_APP_ID", "app-1")
    monkeypatch.setenv("EVOLUS_AGENT_MODEL", "app-1:agent-1")
    monkeypatch.setenv("EVOLUS_WORKFLOW_ID", "wf-1")
    monkeypatch.setenv("EVOLUS_LIBRARY_ID", "lib-1")
    monkeypatch.setenv("EVOLUS_EXTRACTOR_ID", "ex-1")
    monkeypatch.setenv("EVOLUS_EXTRACTOR_AGENT_ID", "agent-x")
    seen = []
    script = {}

    def handler(req: httpx.Request):
        seen.append(req)
        key = (req.method, req.url.path)
        resp = script.get(key)
        if isinstance(resp, list):
            resp = resp.pop(0)
        return resp or httpx.Response(404, json={"error": "nope"})

    real = httpx.Client
    monkeypatch.setattr(client_mod.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    return EvolusClient(get_settings()), seen, script


def test_headers_and_workflow_execute(live):
    c, seen, script = live
    script[("POST", "/api/v1/workflows/wf-1/execute")] = httpx.Response(202, json={"executionId": "ex-9"})
    script[("GET", "/api/v1/workflows/wf-1/executions/ex-9")] = [httpx.Response(200, json={"status": "Running"}),
                                                                 httpx.Response(200, json={"status": "Completed", "output": '{"title": "t"}'})]
    eid = workflow.execute(c, "INPUT", {"asset_id": "A", "route": "supervisor"}, "https://pub/callback/evolus")
    assert eid == "ex-9"
    req = seen[0]
    assert req.headers["X-ApiKey"] == "test-key" and req.headers["X-Application-Id"] == "app-1"
    body = req.content.decode()
    assert 'name="input"' in body and "INPUT" in body and 'name="parameters"' in body and 'name="callbackUrl"' in body
    res = workflow.wait(c, eid, None, timeout_s=5, interval_s=0)
    assert json.loads(workflow.output_text(res)) == {"title": "t"}


def test_workflow_409_and_retry(live):
    c, seen, script = live
    script[("POST", "/api/v1/workflows/wf-1/execute")] = httpx.Response(409, json={"error": "disabled"})
    with pytest.raises(EvolusError) as e:
        workflow.execute(c, "x", {})
    assert e.value.status == 409
    script[("GET", "/api/v1/ai-tools")] = [httpx.Response(500), httpx.Response(200, json=[{"slug": "minutes"}])]
    assert tools.list_tools(c) == [{"slug": "minutes"}]  # one retry on 5xx


def test_chat_extractor_tools_library(live):
    c, seen, script = live
    script[("POST", "/v1/chat/completions")] = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}, headers={"X-Conversation-Id": "conv-1"})
    content, conv = agent.chat(c, [{"role": "user", "content": "hi"}])
    assert content == "{}" and conv == "conv-1"
    assert seen[-1].headers["Authorization"] == "Bearer test-key"
    assert json.loads(seen[-1].content)["model"] == "app-1:agent-1"

    script[("POST", "/api/v1/agents/agent-x/json/ex-1/ask-file")] = httpx.Response(200, json={"result": {"technician": "K"}})
    assert extractor.ask_file(c, b"img", "f.jpg")["result"] == {"technician": "K"}
    assert seen[-1].url.params["useWebhook"] == "false"

    script[("POST", "/api/v1/ai-tools/minutes/run-media")] = httpx.Response(200, json={"sourceText": "hello", "output": "m"})
    assert tools.run_media(c, "minutes", b"aud", "v.m4a")["sourceText"] == "hello"

    script[("POST", "/api/v1/memories/lib-1/documents")] = httpx.Response(200, json={"id": "d1"})
    assert library.upload_document(c, "WO-1.md", b"# x")["id"] == "d1"
    assert seen[-1].url.params["language"] == "en"


def test_mock_mode_without_key(monkeypatch):
    monkeypatch.delenv("EVOLUS_API_KEY", raising=False)
    c = EvolusClient(get_settings())
    assert c.mock
    eid = workflow.execute(c, "x", {}, mock_output={"title": "m"})
    assert workflow.get_execution(c, eid)["status"] == "Completed"
    assert extractor.ask_file(c, b"Technician: Ama\nLOTO removed by: Ama\n", "f.pdf")["result"]["loto_removed_by"] == "Ama"
