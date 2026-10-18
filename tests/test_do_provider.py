"""Primary provider (OpenAI-compatible inference API, MODEL_ACCESS_KEY) with mocked HTTP. No network."""
import json

import httpx
import pytest

from truefield.config import get_settings
from truefield.evolus.client import EvolusClient
from truefield.llm import client as llm_client
from truefield.llm.diagnose import diagnose
from tests.test_llm_layer import EVENT, good

KEY = "test-access-key-not-real"


@pytest.fixture
def do_env(monkeypatch):
    monkeypatch.delenv("MOCK", raising=False)  # HTTP is mocked below; exercise the real provider path
    monkeypatch.setenv("MODEL_ACCESS_KEY", KEY)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    llm_client._NO_RESPONSE_FORMAT.clear()
    seen, replies = [], []

    def handler(req: httpx.Request):
        seen.append(req)
        r = replies.pop(0)
        return r(req) if callable(r) else r

    real = httpx.Client
    monkeypatch.setattr(llm_client.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen, replies


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_defaults_and_auto_selection(monkeypatch):
    monkeypatch.setenv("MOCK", "1")
    monkeypatch.setenv("MODEL_ACCESS_KEY", KEY)
    assert get_settings().llm_mode == "mock"  # MOCK=1 forces mock even with a key
    monkeypatch.delenv("MOCK")
    monkeypatch.delenv("MODEL_ACCESS_KEY")
    s = get_settings()
    assert s.llm_mode == "mock"  # no key -> mock
    assert s.model_base_url == "https://inference.do-ai.run/v1" and s.model_name == "gpt-4o"
    monkeypatch.setenv("MODEL_ACCESS_KEY", KEY)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert get_settings().llm_mode == "do"
    monkeypatch.setenv("EVOLUS_APP_ID", "app")
    monkeypatch.setenv("EVOLUS_AGENT_ID", "ag")
    s = get_settings()
    assert s.evolus_agent_model == "app:ag" and s.evolus_extractor_agent_id == "ag"  # derived


def test_request_shape_and_diagnosis(do_env):
    seen, replies = do_env
    replies.append(ok(good()))
    s = get_settings()
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "diagnosed" and dx["_meta"]["provider"] == "do" and dx["_meta"]["valid_first_try"]
    req = seen[0]
    assert str(req.url) == "https://inference.do-ai.run/v1/chat/completions"
    assert req.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "gpt-4o"
    assert [m["role"] for m in body["messages"]] == ["developer", "user"]
    assert "Truefield Diagnostician" in body["messages"][0]["content"]
    assert body["response_format"] == {"type": "json_object"}


def test_response_format_rejected_falls_back(do_env):
    seen, replies = do_env
    replies += [httpx.Response(400, json={"error": {"message": "Unsupported parameter: response_format"}}), ok(good()), ok(good())]
    s = get_settings()
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "diagnosed"
    assert "response_format" in json.loads(seen[0].content) and "response_format" not in json.loads(seen[1].content)
    diagnose(s, EvolusClient(s), EVENT)  # remembered: no more failed attempts
    assert len(seen) == 3 and "response_format" not in json.loads(seen[2].content)


def test_invalid_then_repair_then_needs_human(do_env):
    seen, replies = do_env
    replies += [ok("not json"), ok('{"event_id": "EV-x"}')]
    s = get_settings()
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "needs_human" and len(seen) == 2
    assert "failed validation" in json.loads(seen[1].content)["messages"][-1]["content"]


def test_hallucinated_citation_filtered(do_env):
    _, replies = do_env
    d = json.loads(good())
    d["citations"].append({"doc": "Fictional Manual", "locator": "p1", "quote": "q"})
    replies.append(ok(json.dumps(d)))
    s = get_settings()
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert [c["doc"] for c in dx["citations"]] == ["asset-PUMP-07.md"] and dx["_meta"]["dropped_citations"] == ["Fictional Manual"]


def test_retry_on_5xx_then_offline(do_env):
    seen, replies = do_env
    replies += [httpx.Response(503), ok(good())]
    s = get_settings()
    assert diagnose(s, EvolusClient(s), EVENT)["status"] == "diagnosed" and len(seen) == 2
    replies += [httpx.Response(500), httpx.Response(500)]
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "needs_human" and dx["_meta"].get("model_offline")


def test_auth_error_needs_human_without_leaking_key(do_env):
    _, replies = do_env
    replies.append(httpx.Response(401, json={"error": "bad key"}))
    s = get_settings()
    dx = diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "needs_human" and KEY not in json.dumps(dx)


def test_pipeline_drafts_wo_with_primary_llm(do_env, tmp_path, monkeypatch):
    """No Evolus workflow configured + live LLM -> WO drafted by the LLM; LOTO merged; uncited step -> supervisor."""
    seen, replies = do_env
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "ae.pt"))
    monkeypatch.setenv("AUTO_PIPELINE", "0")
    from truefield.pipeline import Truefield
    t = Truefield(get_settings())
    t.replay.run_id = "t"
    ev = None
    for i in range(int(t.ds.n * 0.6), t.ds.n):
        t.step(i)
        evs = t.store.list_events()
        if evs:
            ev = evs[0]
            break
    assert ev is not None
    dx = json.loads(good()); dx["event_id"] = ev["event_id"]
    dx["parts"] = [{"part_no": "BRG-ZA2115", "desc": "bearing", "qty": 1}]
    wo_llm = {"title": "BRG-1 outer race on PUMP-07", "steps": [{"text": "Replace bearing", "citation": "WO-0003"}, {"text": "Uncited step"}],
              "loto": [], "parts": [{"part_no": "BRG-ZA2115", "qty": 1}, {"part_no": "INVENTED-1", "qty": 9}]}
    replies += [ok(json.dumps(dx)), ok(json.dumps(wo_llm))]
    wo = t.draft_work_order(ev["event_id"])
    assert wo["drafted_by"] == "agent_direct:do"
    assert [p["part_no"] for p in wo["parts"]] == ["BRG-ZA2115"]
    assert {x["energy_source"] for x in wo["loto"]} >= {"electrical", "mechanical"}
    assert "R-ROUTE-1:uncited_step" in wo["approval"]["reasons"] and wo["status"] == "pending_approval"
    assert any("LOTO" in g for g in wo["guardrails"]) and any("parts" in g for g in wo["guardrails"])
    roles = [m["role"] for m in json.loads(seen[-1].content)["messages"]]
    assert roles == ["developer", "user"]


def test_fail_fast_when_do_forced_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("MOCK", raising=False)
    monkeypatch.delenv("MODEL_ACCESS_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "do")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from truefield.pipeline import Truefield
    with pytest.raises(RuntimeError, match="MODEL_ACCESS_KEY"):
        Truefield(get_settings())
