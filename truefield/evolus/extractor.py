"""Document extractor "Close-out form" (path shape verified):
POST /agents/{agentId}/json/{extractorId}/ask-file?useWebhook=false with file=@closeout.jpg."""
from __future__ import annotations

import json
import re

from truefield.evolus.client import EvolusClient

SAMPLE_FORM = {
    "technician": "Kwame Asante",
    "root_cause_confirmed": "Outer-race spalling on BRG-1",
    "matches_diagnosis": True,
    "actions_done": ["Applied LOTO per LOTO-PUMP-07", "Replaced bearing BRG-ZA2115", "Flushed and refilled oil"],
    "parts_used": [{"part_no": "BRG-ZA2115", "qty": 1}, {"part_no": "OIL-ISO68-5L", "qty": 2}],
    "actual_downtime_minutes": 95,
    "loto_removed_by": "Kwame Asante",
    "follow_up": "Re-check vibration after 24 h",
}


def parse_form_text(text: str) -> dict:
    """Parse 'Field: value' lines (used by mock mode and as a fallback for text-like uploads)."""
    out: dict = {}
    keymap = {"technician": "technician", "root cause": "root_cause_confirmed", "root_cause_confirmed": "root_cause_confirmed",
              "actions": "actions_done", "actions done": "actions_done", "parts": "parts_used", "parts used": "parts_used",
              "downtime": "actual_downtime_minutes", "actual downtime minutes": "actual_downtime_minutes",
              "loto removed by": "loto_removed_by", "locks removed by": "loto_removed_by", "follow up": "follow_up",
              "follow-up": "follow_up", "matches diagnosis": "matches_diagnosis"}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower().replace("_", " ")
        v = v.strip()
        key = keymap.get(k) or keymap.get(k.replace(" ", "_"))
        if not key or not v:
            continue
        if key == "actions_done":
            out[key] = [a.strip() for a in re.split(r";|\|", v) if a.strip()]
        elif key == "parts_used":
            parts = []
            for p in re.split(r";|,", v):
                m = re.match(r"\s*([A-Z0-9\-]+)\s*x\s*(\d+)", p.strip(), re.I)
                if m:
                    parts.append({"part_no": m.group(1), "qty": int(m.group(2))})
            out[key] = parts
        elif key == "actual_downtime_minutes":
            m = re.search(r"\d+", v)
            out[key] = int(m.group()) if m else None
        elif key == "matches_diagnosis":
            out[key] = v.lower() in ("yes", "true", "y")
        else:
            out[key] = v
    return out


def ask_file(c: EvolusClient, content: bytes, filename: str, mime: str = "application/octet-stream") -> dict:
    agent = c.s.evolus_extractor_agent_id or c.s.evolus_agent_id
    ext = c.s.evolus_extractor_id
    path = f"/agents/{agent}/json/{ext}/ask-file?useWebhook=false"
    if c.require("extractor.ask_file", path, agent, ext) == "mock":
        c.audit_mock("extractor.ask_file", path, filename)
        try:
            parsed = parse_form_text(content.decode("utf-8"))
        except UnicodeDecodeError:
            parsed = {}
        return {"mock": True, "result": parsed or dict(SAMPLE_FORM)}
    # TODO: confirm in workspace API reference - which agent ID owns the extractor (EVOLUS_EXTRACTOR_AGENT_ID)
    # and the response envelope (JSON object directly vs {"result": {...}} / {"answer": "..."}).
    r = c.request("extractor.ask_file", "POST", path, files={"file": (filename, content, mime)})
    data = c.json_or_raise(r, "extractor.ask_file")
    for k in ("result", "data", "json", "answer", "output"):
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, dict):
            return {"result": v, "raw": data}
        if isinstance(v, str):
            try:
                return {"result": json.loads(v), "raw": data}
            except json.JSONDecodeError:
                pass
    return {"result": data if isinstance(data, dict) else {}, "raw": data}
