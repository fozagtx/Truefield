"""Diagnosis (FR-5): send AnomalyEvent + rules to the agent, validate the Diagnosis JSON,
retry once with the validation errors, else status=needs_human. Citations must name a library doc."""
from __future__ import annotations

import json
import re
from pathlib import Path

from torque.config import REPO_ROOT, Settings
from torque.evolus.client import EvolusClient
from torque.llm.client import chat, extract_json, prompt
from torque.schemas import errors as schema_errors

KB_DIR = REPO_ROOT / "kb"
EXTRA_DOCS = ["OSHA 29 CFR 1910.147", "osha-1910-147"]


def library_docs(s: Settings) -> list[str]:
    docs = [p.name for p in sorted(KB_DIR.glob("*.md")) if p.name != "README.md"] + EXTRA_DOCS
    learned = s.data_dir / "kb_learned"
    if learned.exists():
        docs += [p.name for p in sorted(learned.glob("*.md"))]
    return docs


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower().rsplit(".", 1)[0] if x.lower().endswith((".md", ".pdf", ".txt")) else x.lower())


def citation_valid(doc: str, docs: list[str]) -> bool:
    d = _norm(doc)
    if not d:
        return False
    return any(_norm(k) in d or d in _norm(k) for k in docs)


def learned_wos_for_asset(s: Settings, asset_id: str) -> list[tuple[str, str]]:
    learned = s.data_dir / "kb_learned"
    out = []
    if learned.exists():
        for p in sorted(learned.glob("WO-*.md")):
            txt = p.read_text(encoding="utf-8")
            if asset_id.split("/")[0] in txt:
                out.append((p.name, txt))
    return out


# ---------- mock diagnostician (deterministic, uses the same kb docs) ----------
def make_mock(s: Settings, event: dict):
    def _mock(messages):
        r = event["rules"]
        z = event.get("z", {})
        f = event["features"]
        hyp = r["fault_hypothesis"]
        cites = []
        if hyp == "outer_race":
            conf = 0.82 if z.get("env_bpfo", 0) >= 8 else 0.66
            cause = "Outer-race defect (spalling) on drive-end bearing BRG-1"
            cites = [
                {"doc": "asset-PUMP-07.md", "locator": "§2.1 Envelope analysis", "quote": "A rising peak in the envelope spectrum at BPFO (236 Hz) and its harmonics, together with rising kurtosis, indicates an outer-race defect such as spalling."},
                {"doc": "past-work-orders.md", "locator": "WO-0003", "quote": "Planned bearing swap during weekend stop; outer race spalled 8 mm"},
                {"doc": "loto-PUMP-07.md", "locator": "Step 3", "quote": "Open the motor breaker at MCC-2 bucket 3B. Apply personal lock and tag. Verify zero energy by attempting a try-start at the local push-button station."},
            ]
            parts = [{"part_no": "BRG-ZA2115", "desc": "Rexnord ZA-2115 double-row bearing", "qty": 1},
                     {"part_no": "OIL-ISO68-5L", "desc": "ISO VG 68 turbine oil, 5 L", "qty": 2}]
            actions = ["Apply LOTO per LOTO-PUMP-07", "Plan bearing replacement at next planned stop (within 24 h)",
                       "Replace bearing BRG-ZA2115 and flush oil circuit", "Re-check vibration after 24 h"]
            status, minutes = "diagnosed", 95
        elif hyp == "lubrication":
            conf, cause, status, minutes = 0.62, "Lubrication starvation on BRG-1", "diagnosed", 40
            cites = [{"doc": "past-work-orders.md", "locator": "WO-0005", "quote": "Cleared blocked oil filter on OP-07, topped up oil"},
                     {"doc": "asset-PUMP-07.md", "locator": "§4 Lubrication spec", "quote": "Above 70 °C investigate lubrication; above 90 °C stop the machine."}]
            parts = [{"part_no": "OIL-ISO68-5L", "desc": "ISO VG 68 turbine oil, 5 L", "qty": 1}]
            actions = ["Apply LOTO per LOTO-PUMP-07", "Check oil filter and flow on OP-07", "Top up oil to level"]
        elif hyp in ("inner_race", "rolling_element", "cage"):
            conf, cause, status, minutes = 0.6, f"{hyp.replace('_', ' ').title()} defect on BRG-1", "diagnosed", None
            cites = [{"doc": "asset-PUMP-07.md", "locator": "§2 Fault frequencies", "quote": "BPFO 236 outer-race; BPFI 297 inner-race; BSF 278 rolling-element; FTF 15 cage"}]
            parts = []
            actions = ["Apply LOTO per LOTO-PUMP-07", "Inspect bearing at next stop"]
        else:
            conf, cause, status, minutes = 0.3, "Unclassified anomaly; no fault-band evidence", "insufficient_evidence", None
            cites = [{"doc": "asset-PUMP-07.md", "locator": "§2 Fault frequencies", "quote": "BPFO 236 outer-race; BPFI 297 inner-race"}]
            parts = []
            actions = ["Escalate to supervisor for manual inspection"]
        learned = learned_wos_for_asset(s, event["asset_id"])
        if learned and status == "diagnosed":
            name, _ = learned[-1]
            cites.append({"doc": name, "locator": name.rsplit(".", 1)[0], "quote": "Learned from closed work order: root cause and repair recorded at close-out."})
            conf = min(0.95, conf + 0.05)
        dx = {
            "event_id": event["event_id"],
            "status": status,
            "likely_cause": cause,
            "component": "BRG-1 (Rexnord ZA-2115)",
            "confidence": round(conf, 2),
            "agrees_with_rules": True,
            "evidence": [f"env_bpfo z={z.get('env_bpfo', 0):.1f} (value {f['env_bpfo']:.4g})",
                         f"kurtosis={f['kurtosis']:.2f} (z={z.get('kurtosis', 0):.1f})",
                         f"rms={f['rms']:.4f} g (z={z.get('rms', 0):.1f})",
                         f"AE score {event['score']:.2f} > threshold {event['threshold']:.2f}"],
            "citations": cites,
            "recommended_actions": actions,
            "parts": parts,
            "est_repair_minutes": minutes,
            "uncertainties": ["Temperature channel is synthetic in this demo",
                              "Defect size cannot be measured from vibration alone; confirm visually at teardown"],
        }
        # Guardrail demo: the mock 'model' tries to downgrade severity; the merge layer must ignore it.
        if s.mock_guardrail_demo and r["severity"] in ("alarm", "critical"):
            dx["suggested_severity"] = "warning"
        return json.dumps(dx)

    return _mock


def build_messages(s: Settings, event: dict, question: str | None = None) -> list[dict]:
    payload = {"event": {k: v for k, v in event.items() if k not in ("created_at",)}, "library_documents": library_docs(s)}
    if question:
        payload["question"] = question
    user = json.dumps(payload, default=str)
    if s.llm_mode == "evolus":
        return [{"role": "user", "content": user}]  # system prompt lives in the Evolus agent settings
    return [{"role": "system", "content": prompt("diagnostician")}, {"role": "user", "content": user}]


def validate(dx, event_id: str, docs: list[str]) -> tuple[list[str], list[str]]:
    """Returns (schema errors, dropped citation docs)."""
    if not isinstance(dx, dict):
        return ["<root>: not an object"], []
    errs = schema_errors("diagnosis", dx)
    if dx.get("event_id") != event_id:
        errs.append(f"event_id must be {event_id}")
    dropped = [c.get("doc", "") for c in dx.get("citations", []) if isinstance(c, dict) and not citation_valid(c.get("doc", ""), docs)]
    return errs, dropped


def diagnose(s: Settings, ec: EvolusClient, event: dict, store=None, question: str | None = None) -> dict:
    docs = library_docs(s)
    messages = build_messages(s, event, question)
    mock_fn = make_mock(s, event)
    meta = {"attempts": 0, "valid_first_try": False, "provider": s.llm_mode, "cached": False, "latency_ms": 0.0, "dropped_citations": []}
    dx = None
    last_err: list[str] = []
    for attempt in range(2):
        meta["attempts"] = attempt + 1
        try:
            res = chat(s, ec, messages, "diagnosis", mock_fn, store)
        except Exception as e:  # model offline -> rules-only, routed to supervisor
            last_err = [f"model call failed: {e}"]
            meta["model_offline"] = True
            break
        meta.update(provider=res["provider"], cached=res["cached"], latency_ms=meta["latency_ms"] + res["latency_ms"],
                    conversation_id=res.get("conversation_id"))
        try:
            cand = extract_json(res["content"])
        except (ValueError, json.JSONDecodeError) as e:
            last_err = [f"invalid JSON: {e}"]
            cand = None
        if cand is not None:
            errs, dropped = validate(cand, event["event_id"], docs)
            if not errs:
                if dropped:
                    cand["citations"] = [c for c in cand["citations"] if c.get("doc") not in dropped]
                    meta["dropped_citations"] = dropped
                    cand.setdefault("uncertainties", []).append(f"Dropped citations to unknown documents: {dropped}")
                    if not cand["citations"]:
                        cand["status"] = "needs_human"
                dx = cand
                meta["valid_first_try"] = attempt == 0
                break
            last_err = errs
        messages = messages + [{"role": "assistant", "content": res["content"]},
                               {"role": "user", "content": "Your output failed validation: " + "; ".join(last_err[:8]) + ". Return ONLY corrected Diagnosis JSON."}]
    if dx is None:
        dx = {
            "event_id": event["event_id"], "status": "needs_human",
            "likely_cause": f"Rules-only summary: {event['rules']['fault_hypothesis']} ({event['rules']['severity']})",
            "component": "BRG-1", "confidence": 0.0, "agrees_with_rules": True,
            "evidence": [f"{k}={v:.4g}" for k, v in event["features"].items()],
            "citations": [], "recommended_actions": ["Escalate to supervisor for manual inspection"],
            "parts": [], "est_repair_minutes": None, "uncertainties": ["Diagnosis model output unavailable or invalid: " + "; ".join(last_err[:3])],
        }
    dx["_meta"] = meta
    return dx


def map_voice_transcript(s: Settings, ec: EvolusClient, transcript: str, store=None) -> dict:
    """§13.3: map transcript -> CloseOut fields. Uses the LLM when configured, else a deterministic parser."""
    def _mock(_messages):
        return json.dumps(parse_transcript(transcript))
    msgs = [{"role": "system", "content": prompt("voice_map")}, {"role": "user", "content": transcript}]
    if s.llm_mode == "evolus":
        msgs = [{"role": "user", "content": prompt("voice_map") + "\n\nTRANSCRIPT:\n" + transcript}]
    try:
        res = chat(s, ec, msgs, "voice_map", _mock, store)
        out = extract_json(res["content"])
        return out if isinstance(out, dict) else parse_transcript(transcript)
    except Exception:
        return parse_transcript(transcript)


def parse_transcript(t: str) -> dict:
    """Regex fallback. Never guesses who removed the locks."""
    out: dict = {}
    m = re.search(r"technician[:,]?\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", t, re.I)
    if m:
        out["technician"] = m.group(1).strip()
    m = re.search(r"locks?\s+(?:were\s+)?removed\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", t, re.I)
    out["loto_removed_by"] = m.group(1).strip() if m else None
    m = re.search(r"(outer[- ]race[^.,;]*|inner[- ]race[^.,;]*|cage[^.,;]*|lubrication[^.,;]*)", t, re.I)
    if m:
        out["root_cause_confirmed"] = m.group(1).strip()
    m = re.search(r"(\d+)\s*min", t, re.I)
    if m:
        out["actual_downtime_minutes"] = int(m.group(1))
    acts = [a.strip() for a in re.findall(r"((?:replaced|flushed|cleaned|inspected|aligned|topped up)[^.,;]*)", t, re.I)]
    if acts:
        out["actions_done"] = acts
    parts = []
    for qty, pn in re.findall(r"(one|two|three|\d+)\s+([A-Z]{2,}[A-Z0-9\-]+)", t):
        q = {"one": 1, "two": 2, "three": 3}[qty.lower()] if qty.isalpha() else int(qty)
        parts.append({"part_no": pn, "qty": q})
    if parts:
        out["parts_used"] = parts
    out["matches_diagnosis"] = bool(re.search(r"matches the diagnosis", t, re.I)) or None
    m = re.search(r"follow[- ]up[:,]?\s+([^.]*)", t, re.I)
    if m:
        out["follow_up"] = m.group(1).strip()
    return out
