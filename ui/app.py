"""Torque Streamlit UI: Technician / Supervisor / Manager / Bench tabs."""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("TORQUE_API_URL", "http://localhost:8080").rstrip("/")
st.set_page_config(page_title="Torque: Maintenance Copilot", page_icon="🔧", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"detail": r.text}
    except requests.RequestException as e:
        return 0, {"detail": f"API unreachable at {API}: {e}"}


SEV_COLOR = {"watch": "#3b82f6", "warning": "#f59e0b", "alarm": "#ef4444", "critical": "#7f1d1d", "normal": "#10b981"}

code, health = api("GET", "/health")
if code != 200:
    st.error(health.get("detail", "API not ready"))
    st.stop()
modes = health["modes"]

st.markdown("## 🔧 Torque: maintenance copilot")
st.caption("Turns a vibration spike into a safe, signed-off work order in under two minutes. AMD × Gemma × Evolus.")
b1, b2, b3, b4 = st.columns(4)
b1.info(f"**Data:** {modes['data_badge']}")
b2.info(f"**Anomaly model:** ae-v1 on {modes['anomaly_device']}")
llm_label = {"mock": "MOCK (no endpoint configured)", "vllm": "Gemma on vLLM (AMD)", "evolus": "Evolus agent → Gemma on vLLM (AMD)"}
b3.info(f"**Diagnosis model:** {llm_label.get(modes['llm'].split()[0], modes['llm'])}{' · cached' if 'cached' in modes['llm'] else ''}")
b4.info(f"**Evolus:** {modes['evolus']} (workflow {modes['workflow']}, extractor {modes['extractor']}, library {modes['library']})")

with st.sidebar:
    st.markdown("### Role")
    role = st.radio("Role switcher", ["Kwame · Technician", "Giulia · Supervisor", "Mensah · Plant manager"], label_visibility="collapsed")
    st.markdown("### Replay (NASA IMS Set 2)")
    speed = st.slider("Snapshots per second", 1, 50, 5)
    start_frac = st.slider("Start at (fraction of record)", 0.0, 0.95, 0.55, 0.05)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", width="stretch"):
        api("POST", "/replay/start", params={"speed": speed, "start_frac": start_frac})
    if c2.button("■ Stop", width="stretch"):
        api("POST", "/replay/stop")
    if st.button("Reset demo state", width="stretch"):
        api("POST", "/reset")
    live = st.toggle("Live refresh (2 s)", value=os.environ.get("TORQUE_UI_LIVE", "1") == "1")
    _, rs = api("GET", "/replay/status")
    st.progress(min(1.0, (rs["idx"] + 1) / rs["total"]), text=f"snapshot {rs['idx'] + 1}/{rs['total']} {'(running)' if rs['running'] else ''}")
    st.caption(f"API: {API}")

tabs = st.tabs(["Technician", "Supervisor", "Manager", "Bench"])
default_tab = {"K": 0, "G": 1, "M": 2}[role[0]]
if default_tab:
    st.caption(f"Role selected: {role}. Use the **{['Technician', 'Supervisor', 'Manager'][default_tab]}** tab.")


def confidence_bar(c: float):
    color, label = ("#ef4444", "Needs review") if c < 0.5 else (("#f59e0b", "Moderate") if c < 0.75 else ("#10b981", "High"))
    st.markdown(f"""<div style="background:#e5e7eb;border-radius:6px;height:18px;width:100%">
      <div style="background:{color};width:{int(c * 100)}%;height:18px;border-radius:6px;color:white;font-size:12px;padding-left:6px">{c:.2f} · {label}</div></div>""",
                unsafe_allow_html=True)


def chips(cites):
    html = " ".join(f'<span title="{c.get("quote", "").replace(chr(34), "")}" style="background:#eef2ff;border:1px solid #6366f1;border-radius:12px;padding:2px 8px;margin:2px;display:inline-block;font-size:12px">📄 {c["doc"]} · {c["locator"]}</span>' for c in cites)
    st.markdown(html, unsafe_allow_html=True)


def render_wo(wo, key_prefix=""):
    status = wo["status"]
    badge = {"pending_approval": "🟠 Awaiting supervisor", "draft": "📝 Draft (technician confirms)", "issued": "🟢 Issued",
             "closed": "✅ Closed", "rejected": "⛔ Rejected"}.get(status, status)
    st.markdown(f"#### {wo['wo_id']} · {wo['title']}")
    st.markdown(f"**{badge}** · priority **{wo['priority']}** · severity **{wo.get('severity')}** · drafted by `{wo.get('drafted_by')}`")
    if wo["approval"]["required"]:
        st.warning("Routing reasons: " + ", ".join(wo["approval"]["reasons"]))
    st.markdown("**LOTO (from deterministic rules, cannot be removed):**")
    st.table(pd.DataFrame(wo["loto"]))
    st.markdown("**Steps:**")
    for s in wo["steps"]:
        st.markdown(f"{s['n']}. {s['text']}" + (f"  · _cite: {s['citation']}_" if s.get("citation") else "  · ⚠️ _uncited_"))
    if wo["parts"]:
        st.markdown("**Parts:** " + ", ".join(f"{p['part_no']} ×{p.get('qty', 1)}" for p in wo["parts"]))
    d = wo["downtime_estimate"]
    st.metric("Estimated downtime cost avoided", f"${d['cost_avoided_usd']:,.0f}", help=d["assumptions"])
    for g in wo.get("guardrails", []):
        st.error("🛡️ " + g)
    st.caption("⚠️ " + wo["safety_notice"])


# ---------------------------------------------------------------- Technician
with tabs[0]:
    st.markdown("**Kwame, night-shift technician** · asset **PUMP-07/BRG-1** (Rexnord ZA-2115, 2000 rpm)")
    _, tel = api("GET", "/telemetry")
    pts = tel.get("points", [])
    if pts:
        df = pd.DataFrame(pts)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["score"], name="AE anomaly score", line=dict(color="#6366f1")))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["threshold"], name="threshold (p99.5 healthy)", line=dict(dash="dash", color="#ef4444")))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp_c"], name="housing temp °C (SYNTHETIC)", yaxis="y2", line=dict(color="#f59e0b", dash="dot")))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(title="score", type="log"),
                          yaxis2=dict(title="°C", overlaying="y", side="right"), legend=dict(orientation="h"),
                          title=tel.get("badge"))
        st.plotly_chart(fig, width="stretch")
        last = pts[-1]
        m = st.columns(5)
        m[0].metric("RMS (g)", f"{last['rms']:.3f}")
        m[1].metric("Kurtosis", f"{last['kurtosis']:.2f}")
        m[2].metric("BPFO 236 Hz z", f"{last['z_env_bpfo']:.1f}")
        m[3].metric("Temp °C (synthetic)", f"{last['temp_c']:.1f}")
        m[4].metric("Severity", last["severity"])
    else:
        st.info("Start the replay from the sidebar.")

    _, events = api("GET", "/events")
    if events:
        ev = events[0]
        r = ev["rules"]
        st.markdown(f"""<div style="border-left:6px solid {SEV_COLOR.get(r['severity'], '#999')};padding:8px 12px;background:#fafafa">
          <b>ALERT {ev['event_id']}</b> · {r['severity'].upper()} · hypothesis <b>{r['fault_hypothesis']}</b> · risk {r['risk_level']} · route {r['route']}<br/>
          data time {ev['ts']} · AE score {ev['score']:.2f} &gt; {ev['threshold']:.2f} ({ev['consecutive']} in a row) · rules fired: {', '.join(r['fired_rules'])}</div>""",
                    unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 3])
        if cols[0].button("Diagnose"):
            api("POST", f"/events/{ev['event_id']}/diagnose")
        if cols[1].button("Draft work order"):
            api("POST", f"/events/{ev['event_id']}/workorder")
        if cols[2].button("False alarm"):
            api("POST", f"/events/{ev['event_id']}/false_alarm", params={"by": "kwame"})
        dx = ev.get("diagnosis")
        if dx:
            st.markdown("### Diagnosis")
            if dx["status"] != "diagnosed":
                st.error(f"Escalated · {dx['status']}")
            st.markdown(f"**{dx['likely_cause']}** · component {dx.get('component', '')} · via `{dx.get('_meta', {}).get('provider')}`")
            confidence_bar(float(dx["confidence"]))
            chips(dx.get("citations", []))
            with st.expander("Evidence, actions, uncertainties"):
                st.write({"evidence": dx.get("evidence"), "recommended_actions": dx.get("recommended_actions"), "uncertainties": dx.get("uncertainties")})
            for g in dx.get("_guardrails", []):
                st.error("🛡️ " + g)
        if ev.get("wo_id"):
            _, wo = api("GET", f"/wo/{ev['wo_id']}")
            render_wo(wo)
            if wo["status"] == "draft" and st.button("Confirm & issue (technician)"):
                c, res = api("POST", f"/wo/{wo['wo_id']}/issue", json={"by": "Kwame Asante"})
                st.write(res if c != 200 else "Issued")
            if wo["status"] in ("issued", "in_progress"):
                st.markdown("### Close-out")
                src = st.radio("Source", ["voice", "photo", "pdf"], horizontal=True)
                up = st.file_uploader("Voice note (m4a/mp3/wav, or .txt transcript in mock mode) / photo / PDF of the close-out form")
                ov = {}
                with st.expander("Fill missing fields (only if prompted)"):
                    ov["technician"] = st.text_input("Technician")
                    ov["loto_removed_by"] = st.text_input("Locks removed by")
                    ov["root_cause_confirmed"] = st.text_input("Root cause confirmed")
                    acts = st.text_input("Actions done (; separated)")
                    if acts:
                        ov["actions_done"] = [a.strip() for a in acts.split(";") if a.strip()]
                if st.button("Submit close-out", disabled=up is None):
                    c, res = api("POST", f"/wo/{wo['wo_id']}/closeout", data={"source": src, "overrides": json.dumps({k: v for k, v in ov.items() if v})},
                                 files={"file": (up.name, up.getvalue(), up.type or "application/octet-stream")})
                    if res.get("closed"):
                        st.success(f"Closed. Asset history updated; {wo['wo_id']}.md uploaded to the knowledge library.")
                    else:
                        st.warning(res.get("prompt") or res.get("detail"))
                        if res.get("transcript"):
                            st.caption("Transcript: " + res["transcript"])
                        st.json(res.get("draft", {}))
            if wo["status"] == "closed":
                st.success("Closed: " + json.dumps({k: wo["closeout"].get(k) for k in ("technician", "root_cause_confirmed", "loto_removed_by", "source")}))
    _, hist = api("GET", "/assets/PUMP-07/BRG-1")
    with st.expander("Asset history"):
        st.dataframe(pd.DataFrame(hist.get("entries", [])), width="stretch")

# ---------------------------------------------------------------- Supervisor
with tabs[1]:
    st.markdown("**Giulia, maintenance supervisor** · approvals")
    _, pend = api("GET", "/wo", params={"status": "pending_approval"})
    if not pend:
        st.info("No work orders awaiting approval.")
    for wo in pend:
        with st.container(border=True):
            _, ev = api("GET", f"/events/{wo['event_id']}")
            render_wo(wo)
            dx = ev.get("diagnosis") or {}
            if dx:
                st.markdown(f"**Diagnosis:** {dx.get('likely_cause')}")
                confidence_bar(float(dx.get("confidence", 0)))
                chips(dx.get("citations", []))
            st.caption("Evidence: " + ", ".join(f"{k}={v:.3g}" for k, v in ev["features"].items()))
            who = st.text_input("Your name", "Giulia Romano", key=f"who-{wo['wo_id']}")
            note = st.text_input("Note", key=f"note-{wo['wo_id']}")
            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Approve", key=f"a-{wo['wo_id']}", width="stretch"):
                api("POST", f"/wo/{wo['wo_id']}/approve", json={"by": who, "decision": "approved", "note": note}); st.rerun()
            if c2.button("✏️ Approve with edit (P1)", key=f"e-{wo['wo_id']}", width="stretch"):
                api("POST", f"/wo/{wo['wo_id']}/approve", json={"by": who, "decision": "edited", "note": note, "edits": {"priority": "P1"}}); st.rerun()
            if c3.button("⛔ Reject", key=f"r-{wo['wo_id']}", width="stretch"):
                api("POST", f"/wo/{wo['wo_id']}/approve", json={"by": who, "decision": "rejected", "note": note}); st.rerun()

# ---------------------------------------------------------------- Manager
with tabs[2]:
    st.markdown("**Mensah, plant manager** · KPIs (computed from logs)")
    full = st.checkbox("Include full-record lead-time evaluation (replays every snapshot offline)", value=True)
    _, k = api("GET", "/kpi", params={"full_eval": full})
    ev_full = k.get("full_record_eval") or {}
    c = st.columns(4)
    c[0].metric("Torque lead time (full record)", f"{ev_full.get('torque_first_alert_lead_h', '–')} h",
                delta=f"+{ev_full.get('hours_gained_vs_naive')} h vs naive" if ev_full.get("hours_gained_vs_naive") is not None else None)
    c[1].metric("Naive 2× RMS alarm lead", f"{ev_full.get('naive_2x_rms_lead_h', '–')} h")
    tw = k["time_to_wo_s"]
    c[2].metric("Time to work order (p50)", f"{tw['p50']} s" if tw["p50"] is not None else "–", help="Manual baseline: " + tw["manual_baseline_min"] + " min")
    c[3].metric("Cost avoided / prevented failure", f"${k['cost_avoided_per_prevented_failure_usd']:,.0f}", help=k["cost_assumptions"])
    c = st.columns(4)
    c[0].metric("Diagnosis JSON validity", k["diagnosis_json_validity_rate"] if k["diagnosis_json_validity_rate"] is not None else "–")
    c[1].metric("Citation rate", k["citation_rate"] if k["citation_rate"] is not None else "–")
    c[2].metric("Guardrail catches", k["guardrail_catches"])
    c[3].metric("Alert → diagnosis p50", f"{k['alert_to_diagnosis_s']['p50']} s" if k["alert_to_diagnosis_s"]["p50"] is not None else "–")
    st.write("Work orders:", k["work_orders"])
    with st.expander("Methodology"):
        st.write(k["methodology"])
        st.write(k["cost_assumptions"])
        st.json(ev_full)
    with st.expander("Audit log (Evolus / vLLM / app calls)"):
        _, aud = api("GET", "/audit", params={"limit": 200})
        st.dataframe(pd.DataFrame(aud), width="stretch")

# ---------------------------------------------------------------- Bench
with tabs[3]:
    _, b = api("GET", "/bench")
    st.markdown("### AMD benchmarks (`bench/results.json`)")
    st.json(b)

if live:
    time.sleep(2)
    st.rerun()
