"""Truefield FastAPI orchestrator: /replay, /telemetry, /events, /wo, /callback/evolus, /closeout, /kpi, /bench."""
from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from truefield import kpi
from truefield.config import REPO_ROOT, Settings, get_settings
from truefield.pipeline import ASSETS, Truefield

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("truefield.api")


class ApproveBody(BaseModel):
    by: str
    decision: str = "approved"
    note: str = ""
    edits: dict | None = None


class IssueBody(BaseModel):
    by: str


class AskBody(BaseModel):
    question: str


def create_app(settings: Settings | None = None, eval_in_background: bool = True) -> FastAPI:
    holder: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        t = Truefield(s)
        holder["t"] = t
        log.info("Truefield ready: %s", t.modes())
        if eval_in_background:
            threading.Thread(target=kpi.evaluate_lead_time, args=(t,), daemon=True).start()
        yield
        t.stop_replay()

    app = FastAPI(title="Truefield API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def T() -> Truefield:
        return holder["t"]

    @app.get("/health")
    def health():
        t = T()
        return {"ok": True, "modes": t.modes(), "dataset": {"label": t.ds.label, "n": t.ds.n, "source": t.ds.source},
                "model": {k: v for k, v in t.scorer.meta.items() if not isinstance(v, list)}}

    @app.get("/assets/{asset_id:path}")
    def asset(asset_id: str):
        if asset_id not in ASSETS:
            raise HTTPException(404, "unknown asset")
        return {"asset_id": asset_id, "meta": {k: v for k, v in ASSETS[asset_id].items() if k in ("type", "rpm", "criticality", "location")},
                "entries": T().store.history(asset_id)}

    # ---------------- replay / telemetry
    @app.post("/replay/start")
    def replay_start(speed: float = 1.0, start: int = 0, start_frac: float | None = None):
        t = T()
        if start_frac is not None:
            start = int(start_frac * t.ds.n)
        t.start_replay(speed=speed, start=start)
        return replay_status()

    @app.post("/replay/stop")
    def replay_stop():
        T().stop_replay()
        return replay_status()

    @app.get("/replay/status")
    def replay_status():
        t = T()
        rp = t.replay
        return {"running": rp.running, "idx": rp.idx, "start": rp.start, "total": t.ds.n, "speed": rp.speed,
                "run_id": rp.run_id, "badge": t.ds.badge, "source": t.ds.source, "last": rp.points[-1] if rp.points else None}

    @app.get("/telemetry")
    def telemetry(since: int = 0):
        pts = T().replay.points
        return {"points": [p for p in pts if p["idx"] >= since], "badge": T().ds.badge}

    @app.post("/reset")
    def reset():
        t = T()
        t.stop_replay()
        t.store.reset()
        t.replay.points = []
        learned = t.s.data_dir / "kb_learned"
        if learned.exists():
            for p in learned.glob("WO-*.md"):
                p.unlink()
        return {"ok": True}

    # ---------------- events / diagnosis
    @app.get("/events")
    def events(limit: int = 50):
        t = T()
        out = []
        for e in t.store.list_events(limit):
            dx = t.store.get_diagnosis(e["event_id"])
            wo = t.store.wo_for_event(e["event_id"])
            out.append({**e, "diagnosis": dx, "wo_id": wo["wo_id"] if wo else None})
        return out

    @app.get("/events/{event_id}")
    def event(event_id: str):
        t = T()
        e = t.store.get_event(event_id)
        if not e:
            raise HTTPException(404, "unknown event")
        return {**e, "diagnosis": t.store.get_diagnosis(event_id), "work_order": t.store.wo_for_event(event_id)}

    @app.post("/events/{event_id}/diagnose")
    def run_diagnose(event_id: str, force: bool = False):
        try:
            return T().run_diagnosis(event_id, force=force)
        except KeyError:
            raise HTTPException(404, "unknown event")

    @app.post("/events/{event_id}/ask")
    def ask(event_id: str, body: AskBody):
        try:
            return T().run_diagnosis(event_id, question=body.question)
        except KeyError:
            raise HTTPException(404, "unknown event")

    @app.post("/events/{event_id}/workorder")
    def make_wo(event_id: str):
        try:
            return T().draft_work_order(event_id)
        except KeyError:
            raise HTTPException(404, "unknown event")

    @app.post("/events/{event_id}/false_alarm")
    def false_alarm(event_id: str, by: str = "technician"):
        t = T()
        e = t.store.get_event(event_id)
        if not e:
            raise HTTPException(404, "unknown event")
        t.store.metric("false_alarm", 1, event_id, by)
        t.store.add_history(e["asset_id"], "anomaly", event_id, f"marked false alarm by {by}")
        return {"ok": True}

    # ---------------- work orders
    @app.get("/wo")
    def list_wo(status: str | None = None):
        return T().store.list_wos(status)

    @app.get("/wo/{wo_id}")
    def get_wo(wo_id: str):
        wo = T().store.get_wo(wo_id)
        if not wo:
            raise HTTPException(404, "unknown work order")
        return wo

    @app.post("/wo/{wo_id}/approve")
    def approve(wo_id: str, body: ApproveBody):
        try:
            return T().approve(wo_id, body.by, body.decision, body.edits, body.note)
        except KeyError:
            raise HTTPException(404, "unknown work order")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/wo/{wo_id}/issue")
    def issue(wo_id: str, body: IssueBody):
        try:
            return T().issue(wo_id, body.by)
        except KeyError:
            raise HTTPException(404, "unknown work order")
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/wo/{wo_id}/closeout")
    async def closeout(wo_id: str, source: str = Form(...), file: UploadFile = File(...), overrides: str = Form("{}")):
        content = await file.read()
        try:
            ov = json.loads(overrides or "{}")
        except json.JSONDecodeError:
            raise HTTPException(400, "overrides must be JSON")
        try:
            res = T().closeout(wo_id, source, file.filename or "upload", content, file.content_type or "", ov)
        except KeyError:
            raise HTTPException(404, "unknown work order")
        except PermissionError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return res

    # ---------------- Evolus workflow callback
    @app.post("/callback/evolus")
    async def callback(request: Request, x_truefield_key: str | None = Header(default=None)):
        t = T()
        if t.s.callback_key and x_truefield_key != t.s.callback_key:
            raise HTTPException(401, "bad callback key")
        try:
            data = await request.json()
        except Exception:
            data = {"raw": (await request.body()).decode("utf-8", "replace")}
        eid = data.get("executionId") or data.get("execution_id") or data.get("id")
        t.store.audit("evolus", "workflow.callback", "POST", "/callback/evolus", 200, 0, "live", json.dumps(data)[:500])
        if eid:
            t.store.put_callback(eid, data)
        return {"ok": True}

    # ---------------- KPIs / bench / audit
    @app.get("/kpi")
    def get_kpi(full_eval: bool = False):
        t = T()
        if full_eval:
            kpi.evaluate_lead_time(t)
        return kpi.compute(t)

    @app.get("/bench")
    def bench():
        p = REPO_ROOT / "bench" / "results.json"
        return json.loads(p.read_text()) if p.exists() else {"note": "run bench/ scripts to produce bench/results.json"}

    @app.get("/audit")
    def audit(limit: int = 100):
        return T().store.audit_rows(limit)

    return app


app = create_app()
