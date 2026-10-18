import pytest

from torque.rules.engine import RulesInput, _rms_rising, evaluate, max_severity, merge_invariants, route, sev_rank

ASSET = {"criticality": "B", "rotating": True, "lube_loop": True}


def ev(z=None, **kw):
    base = {"rms": 0, "kurtosis": 0, "env_bpfo": 0, "env_bpfi": 0, "env_bsf": 0, "env_ftf": 0}
    base.update(z or {})
    kw.setdefault("asset", ASSET)
    return evaluate(RulesInput(z=base, **kw))


def test_normal_no_rules():
    o = ev()
    assert o["severity"] == "normal" and o["risk_level"] == "low" and o["route"] == "auto_draft"
    assert o["fault_hypothesis"] == "unknown" and "R-HYP-3" in o["fired_rules"] and o["priority"] == "P3"


def test_watch_from_ae_debounce():
    assert ev(ae_consecutive=2)["severity"] == "normal"
    o = ev(ae_consecutive=3)
    assert o["severity"] == "watch" and "R-SEV-1" in o["fired_rules"] and o["priority"] == "P3"


@pytest.mark.parametrize("z", [{"rms": 3}, {"kurtosis": 4}])
def test_warning(z):
    o = ev(z)
    assert o["severity"] == "warning" and o["risk_level"] == "medium" and o["priority"] == "P2"


@pytest.mark.parametrize("z", [{"rms": 6}, {"env_bpfo": 8}, {"env_bpfi": 8}])
def test_alarm(z):
    o = ev(z)
    assert o["severity"] == "alarm" and o["risk_level"] == "high" and o["route"] == "supervisor" and o["priority"] == "P1"


def test_critical_temperature():
    o = ev(temp_c=91, temp_baseline_c=45)
    assert o["severity"] == "critical" and "R-SEV-4" in o["fired_rules"]
    assert any(s["energy_source"] == "thermal" for s in o["loto"])


def test_critical_rms_rising():
    hist = [(0, 1.0), (1, 1.3), (2, 1.7), (3, 2.2)]
    assert _rms_rising(hist, 0.2, 3)
    assert ev(rms_history=hist)["severity"] == "critical"
    assert not _rms_rising([(0, 1.0), (1, 1.3), (2, 1.4), (3, 2.2)], 0.2, 3)  # one hour below 20%
    assert not _rms_rising([(0, 1.0)], 0.2, 3)
    assert not _rms_rising([(2, 1.0), (3, 2.0)], 0.2, 3)  # not enough history
    assert not _rms_rising([(0, 0.0), (1, 1.3), (2, 1.7), (3, 2.2)], 0.2, 3)


@pytest.mark.parametrize("band,hyp", [("env_bpfo", "outer_race"), ("env_bpfi", "inner_race"), ("env_bsf", "rolling_element"), ("env_ftf", "cage")])
def test_hypothesis_bands(band, hyp):
    assert ev({band: 5})["fault_hypothesis"] == hyp


def test_hypothesis_largest_band_wins():
    assert ev({"env_bpfo": 5, "env_bpfi": 9})["fault_hypothesis"] == "inner_race"


def test_lubrication_and_unknown():
    assert ev(temp_c=62, temp_baseline_c=45)["fault_hypothesis"] == "lubrication"
    assert ev(temp_c=55, temp_baseline_c=45)["fault_hypothesis"] == "unknown"
    assert ev({"env_bpfo": 5}, temp_c=62, temp_baseline_c=45)["fault_hypothesis"] == "outer_race"


def test_criticality_a_is_high_risk():
    o = ev(asset={"criticality": "A", "rotating": True})
    assert o["risk_level"] == "high" and o["route"] == "supervisor"


def test_safety_steps():
    o = ev()
    srcs = [s["energy_source"] for s in o["loto"]]
    assert srcs == ["electrical", "mechanical", "stored"]
    assert all(s.startswith("R-SAFE") for s in o["required_safety_steps"])
    o2 = ev(asset={"criticality": "B", "rotating": False, "lube_loop": False})
    assert o2["loto"] == []
    assert any(s["energy_source"] == "thermal" for s in ev(temp_c=60)["loto"])
    assert not any(s["energy_source"] == "thermal" for s in ev(temp_c=59.9)["loto"])


def test_deterministic():
    a = ev({"env_bpfo": 9, "rms": 4}, temp_c=70, temp_baseline_c=45)
    b = ev({"env_bpfo": 9, "rms": 4}, temp_c=70, temp_baseline_c=45)
    assert a == b


def test_route_reasons():
    rules = {"risk_level": "low"}
    good = {"confidence": 0.9, "status": "diagnosed", "agrees_with_rules": True}
    assert route(rules, good, {"steps": [{"citation": "x"}]}) == ("auto_draft", [])
    r, reasons = route({"risk_level": "high"}, {"confidence": 0.5, "status": "insufficient_evidence", "agrees_with_rules": False},
                       {"steps": [{"citation": None}]})
    assert r == "supervisor"
    assert reasons == ["R-ROUTE-1:high_risk", "R-ROUTE-1:low_confidence", "R-ROUTE-1:insufficient_evidence",
                       "R-ROUTE-1:llm_disagrees", "R-ROUTE-1:uncited_step"]
    assert route(rules, None, None) == ("supervisor", ["R-ROUTE-1:no_diagnosis"])
    assert route(rules, good, None) == ("auto_draft", [])


def test_llm_can_only_escalate():
    rules = {"severity": "alarm", "risk_level": "high"}
    final, g = merge_invariants(rules, {"suggested_severity": "watch", "suggested_risk_level": "low"})
    assert final == {"severity": "alarm", "risk_level": "high"} and len(g) == 2
    final, g = merge_invariants({"severity": "warning", "risk_level": "medium"}, {"suggested_severity": "critical", "suggested_risk_level": "high"})
    assert final == {"severity": "critical", "risk_level": "high"} and g == []
    final, g = merge_invariants(rules, {})
    assert final == {"severity": "alarm", "risk_level": "high"} and g == []


def test_severity_helpers():
    assert max_severity("watch", None, "alarm") == "alarm"
    assert sev_rank("bogus") == 0
