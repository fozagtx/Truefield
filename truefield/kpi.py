"""KPI panel (FR-10): computed from the store and the dataset, never hard-coded."""
from __future__ import annotations

import threading
from datetime import datetime

import numpy as np

from truefield.rules import engine

_eval_cache: dict = {}
_eval_lock = threading.Lock()


def evaluate_lead_time(t) -> dict:
    """Offline full-record evaluation: replay every snapshot through AE debounce + rules (no LLM)."""
    key = (t.ds.label, t.ds.n, t.scorer.threshold)
    with _eval_lock:
        if key in _eval_cache:
            return _eval_cache[key]
    consec, first_watch, first_alarm, naive = 0, None, None, None
    t0 = t.ds.timestamps[0]
    hist = []
    for i in range(t.ds.n):
        r = t.scorer.score_index(i)
        consec = consec + 1 if r["above"] else 0
        hist.append(((t.ds.timestamps[i] - t0).total_seconds() / 3600, r["features"]["rms"]))
        out = engine.evaluate(engine.RulesInput(z=r["z"], ae_consecutive=consec, temp_c=r["temp_c"],
                                                temp_baseline_c=t.scorer.temp_baseline, rms_history=hist[-30:], asset={"rotating": True, "lube_loop": True}))
        if first_watch is None and engine.sev_rank(out["severity"]) >= 1:
            first_watch = i
        if first_alarm is None and engine.sev_rank(out["severity"]) >= 3:
            first_alarm = i
        if naive is None and r["features"]["rms"] >= 2 * t.healthy_rms:
            naive = i
    end = t.ds.end_of_test

    def hours(i):
        return None if i is None else round((end - t.ds.timestamps[i]).total_seconds() / 3600, 2)

    res = {"dataset": t.ds.label, "source": t.ds.source, "end_of_test": end.isoformat(),
           "truefield_first_alert_ts": t.ds.timestamps[first_watch].isoformat() if first_watch is not None else None,
           "truefield_first_alert_lead_h": hours(first_watch), "truefield_first_alarm_lead_h": hours(first_alarm),
           "naive_2x_rms_ts": t.ds.timestamps[naive].isoformat() if naive is not None else None,
           "naive_2x_rms_lead_h": hours(naive)}
    if res["truefield_first_alert_lead_h"] is not None and res["naive_2x_rms_lead_h"] is not None:
        res["hours_gained_vs_naive"] = round(res["truefield_first_alert_lead_h"] - res["naive_2x_rms_lead_h"], 2)
    with _eval_lock:
        _eval_cache[key] = res
    return res


def _p(vals, q):
    return round(float(np.percentile(vals, q)), 2) if vals else None


def compute(t) -> dict:
    st = t.store
    events = st.list_events(1000)
    dxs = st.list_diagnoses()
    tw = [m["value"] for m in st.metrics("time_to_wo_s")]
    adx = [m["value"] for m in st.metrics("alert_to_dx_s")]
    co = [m["value"] for m in st.metrics("closeout_s")]
    session = {}
    if events:
        first = min(events, key=lambda e: e["ts"])
        session["truefield_first_alert_ts"] = first["ts"]
        session["truefield_first_alert_lead_h"] = round((t.ds.end_of_test - datetime.fromisoformat(first["ts"])).total_seconds() / 3600, 2)
    if t.replay.naive_alarm_idx is not None:
        ts = t.ds.timestamps[t.replay.naive_alarm_idx]
        session["naive_2x_rms_ts"] = ts.isoformat()
        session["naive_2x_rms_lead_h"] = round((t.ds.end_of_test - ts).total_seconds() / 3600, 2)
    di = t.downtime_inputs()
    return {
        "data_source": t.ds.badge,
        "session_lead_time": session,
        "full_record_eval": _eval_cache.get((t.ds.label, t.ds.n, t.scorer.threshold)),
        "events": len(events),
        "diagnoses": len(dxs),
        "alert_to_diagnosis_s": {"p50": _p(adx, 50), "p95": _p(adx, 95), "n": len(adx)},
        "time_to_wo_s": {"p50": _p(tw, 50), "p95": _p(tw, 95), "n": len(tw), "manual_baseline_min": "20-30 (stopwatch, measure once)"},
        "closeout_s": {"p50": _p(co, 50), "n": len(co)},
        "diagnosis_json_validity_rate": round(sum(1 for d in dxs if d.get("_meta", {}).get("valid_first_try")) / len(dxs), 3) if dxs else None,
        "citation_rate": round(sum(1 for d in dxs if d.get("citations")) / len(dxs), 3) if dxs else None,
        "guardrail_catches": len(st.metrics("guardrail")),
        "work_orders": {s: len(st.list_wos(s)) for s in ("pending_approval", "draft", "issued", "closed", "rejected")},
        "cost_avoided_per_prevented_failure_usd": di["cost_avoided_usd"],
        "cost_assumptions": di["assumptions"],
        "methodology": ("Lead time = end of test minus first Truefield alert (data time). Naive baseline = first snapshot with RMS >= 2x healthy mean RMS. "
                        "Time to WO = wall-clock from AnomalyEvent creation to WorkOrder stored. Validity = diagnoses valid against the schema on the first try. "
                        "Citation rate = diagnoses with >=1 citation to a known library document. Guardrail catches = blocked severity downgrades, LOTO omissions, unknown parts."),
    }
