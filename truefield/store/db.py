"""SQLite store: events, diagnoses, work orders, close-outs, asset history, audit log, metrics."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, asset_id TEXT, ts TEXT, created_at TEXT, severity TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS diagnoses (event_id TEXT PRIMARY KEY, created_at TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS work_orders (wo_id TEXT PRIMARY KEY, event_id TEXT, asset_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS closeouts (wo_id TEXT PRIMARY KEY, created_at TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id TEXT, ts TEXT, kind TEXT, ref TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, service TEXT, op TEXT, method TEXT, url TEXT, status INTEGER, latency_ms REAL, mode TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, value REAL, ref TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS callbacks (execution_id TEXT PRIMARY KEY, ts TEXT, data TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _exec(self, sql, args=()):
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def _all(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def reset(self):
        with self._lock:
            for t in ("events", "diagnoses", "work_orders", "closeouts", "history", "audit_log", "metrics", "callbacks"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()

    # events
    def add_event(self, ev: dict):
        self._exec("INSERT INTO events VALUES (?,?,?,?,?,?)", (ev["event_id"], ev["asset_id"], ev["ts"], ev["created_at"], ev["rules"]["severity"], json.dumps(ev)))

    def get_event(self, event_id: str) -> dict | None:
        r = self._all("SELECT data FROM events WHERE event_id=?", (event_id,))
        return json.loads(r[0]["data"]) if r else None

    def update_event(self, ev: dict):
        self._exec("UPDATE events SET data=?, severity=? WHERE event_id=?", (json.dumps(ev), ev["rules"]["severity"], ev["event_id"]))

    def list_events(self, limit: int = 100) -> list[dict]:
        return [json.loads(r["data"]) for r in self._all("SELECT data FROM events ORDER BY created_at DESC LIMIT ?", (limit,))]

    def count_events(self) -> int:
        return self._all("SELECT COUNT(*) AS n FROM events")[0]["n"]

    # diagnoses
    def put_diagnosis(self, event_id: str, dx: dict):
        self._exec("INSERT OR REPLACE INTO diagnoses VALUES (?,?,?)", (event_id, now_iso(), json.dumps(dx)))

    def get_diagnosis(self, event_id: str) -> dict | None:
        r = self._all("SELECT data FROM diagnoses WHERE event_id=?", (event_id,))
        return json.loads(r[0]["data"]) if r else None

    def list_diagnoses(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._all("SELECT data FROM diagnoses ORDER BY created_at")]

    # work orders
    def next_wo_id(self) -> str:
        r = self._all("SELECT wo_id FROM work_orders")
        nums = [int(x["wo_id"].split("-")[1]) for x in r if x["wo_id"].startswith("WO-")]
        return f"WO-{(max(nums) + 1 if nums else 101):04d}"  # WO-0001..0012 are the synthetic history

    def put_wo(self, wo: dict):
        ts = now_iso()
        existing = self.get_wo(wo["wo_id"])
        created = existing["_created_at"] if existing else wo.get("_created_at", ts)
        wo["_created_at"] = created
        self._exec("INSERT OR REPLACE INTO work_orders VALUES (?,?,?,?,?,?,?)", (wo["wo_id"], wo["event_id"], wo["asset_id"], wo["status"], created, ts, json.dumps(wo)))

    def get_wo(self, wo_id: str) -> dict | None:
        r = self._all("SELECT data FROM work_orders WHERE wo_id=?", (wo_id,))
        return json.loads(r[0]["data"]) if r else None

    def wo_for_event(self, event_id: str) -> dict | None:
        r = self._all("SELECT data FROM work_orders WHERE event_id=?", (event_id,))
        return json.loads(r[0]["data"]) if r else None

    def open_wo_for_asset(self, asset_id: str) -> dict | None:
        r = self._all("SELECT data FROM work_orders WHERE asset_id=? AND status NOT IN ('closed','rejected') ORDER BY created_at DESC LIMIT 1", (asset_id,))
        return json.loads(r[0]["data"]) if r else None

    def list_wos(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self._all("SELECT data FROM work_orders WHERE status=? ORDER BY created_at DESC", (status,))
        else:
            rows = self._all("SELECT data FROM work_orders ORDER BY created_at DESC")
        return [json.loads(r["data"]) for r in rows]

    # close-outs
    def put_closeout(self, wo_id: str, co: dict):
        self._exec("INSERT OR REPLACE INTO closeouts VALUES (?,?,?)", (wo_id, now_iso(), json.dumps(co)))

    def list_closeouts(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._all("SELECT data FROM closeouts ORDER BY created_at")]

    # history
    def add_history(self, asset_id: str, kind: str, ref: str, summary: str):
        self._exec("INSERT INTO history (asset_id, ts, kind, ref, summary) VALUES (?,?,?,?,?)", (asset_id, now_iso(), kind, ref, summary))

    def history(self, asset_id: str) -> list[dict]:
        return self._all("SELECT ts, kind, ref, summary FROM history WHERE asset_id=? ORDER BY id", (asset_id,))

    # audit
    def audit(self, service, op, method="", url="", status=None, latency_ms=None, mode="live", detail=""):
        self._exec("INSERT INTO audit_log (ts, service, op, method, url, status, latency_ms, mode, detail) VALUES (?,?,?,?,?,?,?,?,?)",
                   (now_iso(), service, op, method, url, status, latency_ms, mode, (detail or "")[:2000]))

    def audit_rows(self, limit: int = 200) -> list[dict]:
        return self._all("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))

    # metrics
    def metric(self, name: str, value: float, ref: str = "", detail: str = ""):
        self._exec("INSERT INTO metrics (ts, name, value, ref, detail) VALUES (?,?,?,?,?)", (now_iso(), name, float(value), ref, detail))

    def metrics(self, name: str) -> list[dict]:
        return self._all("SELECT * FROM metrics WHERE name=? ORDER BY id", (name,))

    # callbacks
    def put_callback(self, execution_id: str, data: dict):
        self._exec("INSERT OR REPLACE INTO callbacks VALUES (?,?,?)", (execution_id, now_iso(), json.dumps(data)))

    def get_callback(self, execution_id: str) -> dict | None:
        r = self._all("SELECT data FROM callbacks WHERE execution_id=?", (execution_id,))
        return json.loads(r[0]["data"]) if r else None
