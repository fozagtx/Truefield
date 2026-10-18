"""Validation + retry + needs_human fallback with a scripted fake model."""
import json

from torque.config import get_settings
from torque.evolus.client import EvolusClient
from torque.llm import diagnose as D

EVENT = {"event_id": "EV-x", "asset_id": "PUMP-07/BRG-1", "ts": "2004-02-18T10:00:00", "score": 5.0, "threshold": 1.0,
         "features": {"rms": 0.2, "peak": 1, "crest": 5, "kurtosis": 8, "env_bpfo": 100, "env_bpfi": 0.1, "env_bsf": 0.1, "env_ftf": 0.1},
         "z": {"env_bpfo": 50, "kurtosis": 20, "rms": 10},
         "rules": {"severity": "alarm", "fault_hypothesis": "outer_race", "risk_level": "high", "route": "supervisor", "fired_rules": [], "required_safety_steps": [], "loto": []}}


def run_with(monkeypatch, replies):
    calls = []

    def fake_chat(s, ec, messages, kind, mock_fn, store=None):
        calls.append(messages)
        return {"content": replies[len(calls) - 1], "provider": "fake", "cached": False, "latency_ms": 1.0}

    monkeypatch.setattr(D, "chat", fake_chat)
    s = get_settings()
    return D.diagnose(s, EvolusClient(s), EVENT), calls


def good():
    return json.dumps({"event_id": "EV-x", "status": "diagnosed", "likely_cause": "outer race", "confidence": 0.8, "evidence": ["e"],
                       "citations": [{"doc": "asset-PUMP-07.md", "locator": "§2", "quote": "q"}], "recommended_actions": ["a"], "uncertainties": []})


def test_valid_first_try(monkeypatch):
    dx, calls = run_with(monkeypatch, [good()])
    assert dx["status"] == "diagnosed" and dx["_meta"]["valid_first_try"] and len(calls) == 1


def test_retry_once_then_ok(monkeypatch):
    dx, calls = run_with(monkeypatch, ["not json at all", good()])
    assert dx["status"] == "diagnosed" and not dx["_meta"]["valid_first_try"] and len(calls) == 2
    assert "failed validation" in calls[1][-1]["content"]


def test_twice_invalid_needs_human(monkeypatch):
    dx, calls = run_with(monkeypatch, ['{"event_id": "EV-x"}', "{}"])
    assert dx["status"] == "needs_human" and dx["confidence"] == 0.0 and len(calls) == 2


def test_only_hallucinated_citations_needs_human(monkeypatch):
    bad = json.loads(good())
    bad["citations"] = [{"doc": "Made-up Handbook", "locator": "p9", "quote": "q"}]
    dx, _ = run_with(monkeypatch, [json.dumps(bad)])
    assert dx["status"] == "needs_human" and dx["citations"] == []


def test_model_offline(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(D, "chat", boom)
    s = get_settings()
    dx = D.diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "needs_human" and dx["_meta"].get("model_offline")


def test_mock_mode_end_to_end():
    s = get_settings()
    dx = D.diagnose(s, EvolusClient(s), EVENT)
    assert dx["status"] == "diagnosed" and dx["citations"] and dx["_meta"]["provider"] == "mock"
