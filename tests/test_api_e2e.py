"""End-to-end in mock mode: replay -> anomaly -> diagnosis -> work order -> approve -> close-out -> history."""
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    d = tmp_path_factory.mktemp("data")
    os.environ["DATA_DIR"] = str(d)
    os.environ["MODEL_PATH"] = str(d / "ae-v1.pt")
    os.environ["AUTO_PIPELINE"] = "0"
    from truefield.api.main import create_app
    from truefield.config import get_settings
    app = create_app(get_settings(), eval_in_background=False)
    with TestClient(app) as c:
        yield c


def wait_for(fn, timeout=60, every=0.2):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("timeout")


def test_full_flow(client):
    h = client.get("/health").json()
    assert h["modes"]["llm"].startswith("mock") and h["modes"]["evolus"] == "mock"
    assert "synthetic" in h["modes"]["data_badge"].lower()

    client.post("/replay/start", params={"speed": 500, "start_frac": 0.5})
    # wait for an alarm-level event
    def alarm():
        evs = client.get("/events").json()
        return next((e for e in evs if e["rules"]["severity"] in ("alarm", "critical")), None)
    ev = wait_for(alarm, timeout=90)
    client.post("/replay/stop")
    assert ev["rules"]["fault_hypothesis"] == "outer_race"
    assert ev["rules"]["route"] == "supervisor"
    assert any(s.startswith("R-SAFE-1") for s in ev["rules"]["required_safety_steps"])

    dx = client.post(f"/events/{ev['event_id']}/diagnose").json()
    assert dx["status"] == "diagnosed" and len(dx["citations"]) >= 1
    assert dx["_final"]["severity"] == ev["rules"]["severity"]  # mock tried to lower it: guardrail
    assert dx["_guardrails"]

    wo = client.post(f"/events/{ev['event_id']}/workorder").json()
    assert wo["status"] == "pending_approval"
    assert "R-ROUTE-1:high_risk" in wo["approval"]["reasons"]
    assert {x["energy_source"] for x in wo["loto"]} >= {"electrical", "mechanical", "stored"}

    r = client.post(f"/wo/{wo['wo_id']}/issue", json={"by": "kwame"})
    assert r.status_code == 403  # cannot bypass supervisor

    r = client.post(f"/wo/{wo['wo_id']}/approve", json={"by": "giulia", "decision": "approved"})
    assert r.status_code == 200 and r.json()["status"] == "issued"

    # voice note without lock remover -> blocked, then provided
    txt = b"Technician Kwame Asante. Replaced the bearing, outer race spalling confirmed. Downtime 90 minutes."
    r = client.post(f"/wo/{wo['wo_id']}/closeout", data={"source": "voice"}, files={"file": ("note.txt", txt, "text/plain")}).json()
    assert r["closed"] is False and "loto_removed_by" in r["missing"]
    r = client.post(f"/wo/{wo['wo_id']}/closeout", data={"source": "voice", "overrides": json.dumps({"loto_removed_by": "Kwame Asante"})},
                    files={"file": ("note.txt", txt, "text/plain")}).json()
    assert r["closed"] is True, r
    assert client.get(f"/wo/{wo['wo_id']}").json()["status"] == "closed"

    hist = client.get("/assets/PUMP-07/BRG-1").json()["entries"]
    kinds = {e["kind"] for e in hist}
    assert {"anomaly", "diagnosis", "work_order", "approval", "closeout"} <= kinds

    # learned from the closed WO on the next diagnosis
    dx2 = client.post(f"/events/{ev['event_id']}/diagnose", params={"force": True}).json()
    assert any(c["doc"] == f"{wo['wo_id']}.md" for c in dx2["citations"])

    k = client.get("/kpi").json()
    assert k["citation_rate"] == 1.0 and k["guardrail_catches"] >= 1
    assert k["time_to_wo_s"]["n"] == 1
    audit = client.get("/audit").json()
    assert any(a["service"] == "evolus" and a["op"] == "workflow.execute" for a in audit)


def test_photo_closeout_form(client):
    # second WO via a fresh replay; photo/pdf path uses the extractor (mock parses form text)
    client.post("/reset")
    client.post("/replay/start", params={"speed": 500, "start_frac": 0.7})
    ev = wait_for(lambda: next((e for e in client.get("/events").json() if e["rules"]["severity"] in ("warning", "alarm", "critical")), None), 90)
    client.post("/replay/stop")
    wo = client.post(f"/events/{ev['event_id']}/workorder").json()
    client.post(f"/wo/{wo['wo_id']}/approve", json={"by": "giulia", "decision": "approved"})
    form = b"Technician: Ama Mensah\nRoot cause: Outer-race spalling\nActions done: Replaced bearing; Flushed oil\nParts used: BRG-ZA2115 x1\nLOTO removed by: Ama Mensah\n"
    r = client.post(f"/wo/{wo['wo_id']}/closeout", data={"source": "pdf"}, files={"file": ("form.pdf", form, "application/pdf")}).json()
    assert r["closed"] is True and r["closeout"]["technician"] == "Ama Mensah"
