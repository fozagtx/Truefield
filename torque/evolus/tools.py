"""AI Tools (verified): GET /ai-tools; POST /ai-tools/{tool}/run, /run-media, /run-file.
Slugs verified: translate, minutes, anonymize. Others (summary, email): TODO confirm via GET /api/v1/ai-tools."""
from __future__ import annotations

import json

from torque.evolus.client import EvolusClient

SAMPLE_TRANSCRIPT = (
    "Close-out for the PUMP-07 job. Technician Kwame Asante. Replaced the drive-end bearing, "
    "outer race spalling confirmed, matches the diagnosis. Used one BRG-ZA2115 bearing and two OIL-ISO68-5L. "
    "Downtime was 95 minutes. Locks removed by Kwame Asante. Follow up: re-check vibration after 24 hours."
)


def list_tools(c: EvolusClient) -> list:
    if c.mock:
        c.audit_mock("tools.list", "/ai-tools")
        return [{"slug": s} for s in ("minutes", "translate", "anonymize")]
    r = c.request("tools.list", "GET", "/ai-tools")
    d = c.json_or_raise(r, "tools.list")
    return d if isinstance(d, list) else d.get("items") or d.get("data") or []


def run(c: EvolusClient, slug: str, text: str, options: dict | None = None) -> dict:
    path = f"/ai-tools/{slug}/run"
    if c.mock:
        c.audit_mock("tools.run", path, slug)
        return {"output": text, "mock": True}
    # TODO: confirm in workspace API reference - body field names ("text"/"input", "options").
    r = c.request("tools.run", "POST", path, json={"text": text, "options": options or {}})
    return c.json_or_raise(r, "tools.run")


def run_media(c: EvolusClient, slug: str, content: bytes, filename: str, mime: str = "audio/mp4", options: dict | None = None) -> dict:
    """For the voice note: returns transcript in `sourceText`, minutes/summary in `output` (per PRD)."""
    path = f"/ai-tools/{slug}/run-media"
    if c.mock:
        c.audit_mock("tools.run_media", path, filename)
        try:
            txt = content.decode("utf-8") if filename.endswith(".txt") else SAMPLE_TRANSCRIPT
        except UnicodeDecodeError:
            txt = SAMPLE_TRANSCRIPT
        return {"sourceText": txt, "output": "Minutes: bearing replaced on PUMP-07; LOTO removed by technician.", "mock": True}
    files = {"file": (filename, content, mime)}
    data = {"options": json.dumps(options)} if options else None
    r = c.request("tools.run_media", "POST", path, files=files, data=data)
    return c.json_or_raise(r, "tools.run_media")


def run_file(c: EvolusClient, slug: str, content: bytes, filename: str, options: dict | None = None) -> bytes | dict:
    path = f"/ai-tools/{slug}/run-file"
    if c.mock:
        c.audit_mock("tools.run_file", path, filename)
        return content
    r = c.request("tools.run_file", "POST", path, files={"file": (filename, content)}, data={"options": json.dumps(options or {})})
    if r.headers.get("content-type", "").startswith("application/json"):
        return c.json_or_raise(r, "tools.run_file")
    return r.content
