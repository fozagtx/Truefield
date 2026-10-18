"""Deterministic rules engine (FR-4, PRD §14). Pure Python, no LLM, same input -> same output."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

SEVERITIES = ["normal", "watch", "warning", "alarm", "critical"]
RISKS = ["low", "medium", "high"]
RULES_PATH = Path(__file__).with_name("rules.yaml")


@lru_cache(maxsize=4)
def load_rules(path: str = str(RULES_PATH)) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sev_rank(s: str | None) -> int:
    return SEVERITIES.index(s) if s in SEVERITIES else 0


def max_severity(*sevs: str | None) -> str:
    return SEVERITIES[max(sev_rank(s) for s in sevs)]


@dataclass
class RulesInput:
    z: dict                       # z-scores per feature name
    ae_consecutive: int = 0       # consecutive snapshots with AE score > threshold
    temp_c: float | None = None
    temp_baseline_c: float | None = None
    rms_history: list = field(default_factory=list)  # [(hours_since_start, rms)], oldest first
    asset: dict = field(default_factory=dict)        # meta: criticality, rotating, lube_loop


def _rms_rising(hist: list, rise: float, hours: int) -> bool:
    """True if rms rose more than `rise` (fraction) in each of the last `hours` one-hour steps."""
    if len(hist) < 2:
        return False
    t_now = hist[-1][0]
    if t_now - hist[0][0] < hours:
        return False

    def rms_at(t):
        best = None
        for th, r in hist:  # last sample at or before t
            if th <= t + 1e-9:
                best = r
        return best

    for k in range(hours):
        newer, older = rms_at(t_now - k), rms_at(t_now - k - 1)
        if older is None or newer is None or older <= 0 or newer / older - 1.0 <= rise:
            return False
    return True


def evaluate(inp: RulesInput, rules: dict | None = None) -> dict:
    r = rules or load_rules()
    z = inp.z
    fired: list[str] = []
    severity = "normal"

    s1 = r["severity"]["R-SEV-1"]
    if inp.ae_consecutive >= s1["ae_consecutive"]:
        severity = max_severity(severity, "watch"); fired.append("R-SEV-1")
    s2 = r["severity"]["R-SEV-2"]
    if z.get("rms", 0) >= s2["z_rms"] or z.get("kurtosis", 0) >= s2["z_kurtosis"]:
        severity = max_severity(severity, "warning"); fired.append("R-SEV-2")
    s3 = r["severity"]["R-SEV-3"]
    if z.get("rms", 0) >= s3["z_rms"] or max(z.get("env_bpfo", 0), z.get("env_bpfi", 0)) >= s3["z_env_fault"]:
        severity = max_severity(severity, "alarm"); fired.append("R-SEV-3")
    s4 = r["severity"]["R-SEV-4"]
    hot = inp.temp_c is not None and inp.temp_c >= s4["temp_c"]
    if hot or _rms_rising(inp.rms_history, s4["rms_rise_per_hour"], s4["rms_rise_hours"]):
        severity = "critical"; fired.append("R-SEV-4")

    h1 = r["hypothesis"]["R-HYP-1"]
    bands = {k: z.get(k, 0.0) for k in h1["bands"]}
    best_band = max(bands, key=lambda k: (bands[k], k))
    if bands[best_band] >= h1["z_min"]:
        hypothesis = h1["bands"][best_band]; fired.append("R-HYP-1")
    elif (inp.temp_c is not None and inp.temp_baseline_c is not None
          and inp.temp_c - inp.temp_baseline_c >= r["hypothesis"]["R-HYP-2"]["temp_rise_c"]):
        hypothesis = "lubrication"; fired.append("R-HYP-2")
    else:
        hypothesis = "unknown"; fired.append("R-HYP-3")

    if severity in ("alarm", "critical") or inp.asset.get("criticality") == "A":
        risk = "high"; fired.append("R-RISK-1")
    elif severity == "warning":
        risk = "medium"; fired.append("R-RISK-2")
    else:
        risk = "low"

    loto: list[dict] = []
    if inp.asset.get("rotating", True):
        loto += [dict(x, rule="R-SAFE-1") for x in r["safety"]["R-SAFE-1"]]; fired.append("R-SAFE-1")
    if inp.asset.get("lube_loop", False):
        loto += [dict(x, rule="R-SAFE-2") for x in r["safety"]["R-SAFE-2"]]; fired.append("R-SAFE-2")
    th = r["safety"]["R-SAFE-3"][0]
    if inp.temp_c is not None and inp.temp_c >= th["min_temp_c"]:
        loto += [{k: v for k, v in x.items() if k != "min_temp_c"} | {"rule": "R-SAFE-3"} for x in r["safety"]["R-SAFE-3"]]
        fired.append("R-SAFE-3")

    pre_route = "supervisor" if risk == "high" else "auto_draft"
    return {
        "severity": severity,
        "fault_hypothesis": hypothesis,
        "risk_level": risk,
        "route": pre_route,
        "fired_rules": fired,
        "required_safety_steps": [f"{s['rule']}/{s['energy_source']}: {s['isolation_point']} (verify: {s['verify']})" for s in loto],
        "loto": loto,
        "priority": r["priority"][severity],
    }


def route(rules_out: dict, dx: dict | None, wo: dict | None, rules: dict | None = None) -> tuple[str, list[str]]:
    """R-ROUTE-1 / R-ROUTE-2. Any reason -> supervisor."""
    r = rules or load_rules()
    reasons = []
    if rules_out.get("risk_level") == "high":
        reasons.append("R-ROUTE-1:high_risk")
    if dx is None:
        reasons.append("R-ROUTE-1:no_diagnosis")
    else:
        if float(dx.get("confidence") or 0.0) < r["routing"]["min_confidence"]:
            reasons.append("R-ROUTE-1:low_confidence")
        if dx.get("status") != "diagnosed":
            reasons.append("R-ROUTE-1:" + str(dx.get("status")))
        if dx.get("agrees_with_rules") is False:
            reasons.append("R-ROUTE-1:llm_disagrees")
    if wo is not None and any(not s.get("citation") for s in wo.get("steps", [])):
        reasons.append("R-ROUTE-1:uncited_step")
    return ("supervisor", reasons) if reasons else ("auto_draft", [])


def merge_invariants(rules_out: dict, dx: dict) -> tuple[dict, list[str]]:
    """final.severity = max(rules, llm); final.risk = max(rules, llm). LLM can only escalate."""
    guardrails = []
    final = {"severity": rules_out["severity"], "risk_level": rules_out["risk_level"]}
    s = dx.get("suggested_severity")
    if s in SEVERITIES:
        if sev_rank(s) < sev_rank(rules_out["severity"]):
            guardrails.append(f"guardrail_triggered: LLM suggested severity '{s}' below rules '{rules_out['severity']}'; ignored")
        final["severity"] = max_severity(rules_out["severity"], s)
    rk = dx.get("suggested_risk_level")
    if rk in RISKS:
        if RISKS.index(rk) < RISKS.index(rules_out["risk_level"]):
            guardrails.append(f"guardrail_triggered: LLM suggested risk '{rk}' below rules '{rules_out['risk_level']}'; ignored")
        else:
            final["risk_level"] = rk
    return final, guardrails
