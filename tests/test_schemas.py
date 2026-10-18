import json

from truefield.llm.diagnose import citation_valid, parse_transcript, validate
from truefield.llm.client import extract_json
from truefield.schemas import errors

GOOD = {"event_id": "EV-1", "status": "diagnosed", "likely_cause": "outer race", "confidence": 0.8, "evidence": ["env_bpfo z=9"],
        "citations": [{"doc": "asset-PUMP-07.md", "locator": "§2", "quote": "BPFO 236 Hz"}], "recommended_actions": ["Replace bearing"],
        "uncertainties": []}
DOCS = ["asset-PUMP-07.md", "past-work-orders.md", "OSHA 29 CFR 1910.147"]


def test_good_diagnosis():
    assert errors("diagnosis", GOOD) == []
    assert validate(GOOD, "EV-1", DOCS) == ([], [])


def test_bad_diagnosis():
    bad = dict(GOOD, confidence=1.5, citations=[], status="maybe")
    errs = errors("diagnosis", bad)
    assert len(errs) == 3
    errs, _ = validate(dict(GOOD), "EV-2", DOCS)
    assert any("event_id" in e for e in errs)
    assert validate("nope", "EV-1", DOCS)[0]


def test_hallucinated_citation_dropped():
    dx = dict(GOOD, citations=GOOD["citations"] + [{"doc": "Imaginary Manual v9", "locator": "p1", "quote": "x"}])
    errs, dropped = validate(dx, "EV-1", DOCS)
    assert errs == [] and dropped == ["Imaginary Manual v9"]
    assert citation_valid("asset-PUMP-07", DOCS) and citation_valid("OSHA 29 CFR 1910.147 (LOTO)", DOCS)
    assert not citation_valid("", DOCS)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}


def test_closeout_and_wo_schema():
    co = {"wo_id": "WO-0101", "completed_at": "2026-10-18T10:00:00Z", "technician": "K", "root_cause_confirmed": "outer race",
          "actions_done": ["replaced"], "loto_removed_by": "K", "source": "voice"}
    assert errors("closeout", co) == []
    assert errors("closeout", dict(co, loto_removed_by=None))
    wo = {"wo_id": "WO-1", "event_id": "E", "asset_id": "A", "status": "pending_approval", "priority": "P1", "title": "t", "steps": [],
          "loto": [], "parts": [], "approval": {}, "downtime_estimate": {}}
    assert any("loto" in e for e in errors("work_order", wo))  # LOTO is mandatory


def test_transcript_never_guesses_lock_remover():
    t = parse_transcript("Technician Kwame Asante. Replaced the bearing, outer race spalling confirmed. Downtime 95 minutes.")
    assert t["loto_removed_by"] is None and t["technician"] == "Kwame Asante" and t["actual_downtime_minutes"] == 95
    t2 = parse_transcript("Used one BRG-ZA2115 and two OIL-ISO68-5L. Locks removed by Kwame Asante.")
    assert t2["loto_removed_by"] == "Kwame Asante"
    assert {"part_no": "OIL-ISO68-5L", "qty": 2} in t2["parts_used"]
