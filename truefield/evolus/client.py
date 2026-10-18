"""Thin Evolus REST/OpenAI-compatible client with timeout, one retry, audit logging and mock mode.

Headers (verified on evolus.ai/developers):
  REST:  X-ApiKey + X-Application-Id
  Chat:  Authorization: Bearer <EVOLUS_API_KEY>
Mock mode is on whenever EVOLUS_API_KEY is empty; every call (live or mock) is written to audit_log.
"""
from __future__ import annotations

import json
import logging
import time

import httpx

from truefield.config import Settings

log = logging.getLogger("truefield.evolus")


class EvolusError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status = status
        self.body = body


class EvolusClient:
    def __init__(self, settings: Settings, store=None):
        self.s = settings
        self.store = store

    @property
    def mock(self) -> bool:
        return not self.s.evolus_enabled

    def rest_headers(self) -> dict:
        h = {"X-ApiKey": self.s.evolus_api_key}
        if self.s.evolus_app_id:
            h["X-Application-Id"] = self.s.evolus_app_id
        return h

    def chat_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.evolus_api_key}"}

    def audit(self, op, method="", url="", status=None, latency_ms=None, mode="live", detail=""):
        log.info("evolus %s %s %s -> %s (%s, %.0f ms)", op, method, url, status, mode, latency_ms or 0)
        if self.store is not None:
            self.store.audit("evolus", op, method, url, status, latency_ms, mode, detail)

    def audit_mock(self, op: str, path: str, detail: str = ""):
        self.audit(op, "MOCK", path, 200, 0.0, "mock", detail)

    def request(self, op: str, method: str, path: str, *, base: str = "rest", **kw) -> httpx.Response:
        """One HTTP call with a timeout and one retry on transport errors / 5xx / 429."""
        root = self.s.evolus_rest_base if base == "rest" else self.s.evolus_chat_base
        url = path if path.startswith("http") else f"{root}{path}"
        headers = {**(self.rest_headers() if base == "rest" else self.chat_headers()), **kw.pop("headers", {})}
        last_exc = None
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.s.http_timeout_s) as c:
                    r = c.request(method, url, headers=headers, **kw)
                ms = (time.perf_counter() - t0) * 1000
                self.audit(op, method, url, r.status_code, ms, "live", r.text[:500])
                if r.status_code in (429,) or r.status_code >= 500:
                    last_exc = EvolusError(f"{op}: HTTP {r.status_code}", r.status_code, r.text)
                    time.sleep(3 if r.status_code == 429 else 1)
                    continue
                return r
            except httpx.HTTPError as e:
                ms = (time.perf_counter() - t0) * 1000
                self.audit(op, method, url, None, ms, "live", f"error: {e!r}")
                last_exc = EvolusError(f"{op}: {e!r}")
        raise last_exc  # type: ignore[misc]

    def json_or_raise(self, r: httpx.Response, op: str):
        if r.status_code >= 400:
            raise EvolusError(f"{op}: HTTP {r.status_code}", r.status_code, r.text)
        try:
            return r.json()
        except json.JSONDecodeError:
            return {"raw": r.text}
