"""Workflow "Draft Work Order" (verified): POST /workflows/{id}/execute (multipart: input, callbackUrl,
callbackMethod, callbackApiKey, callbackApiKeyName) -> 202 {executionId};
GET /workflows/{id}/executions/{executionId} until Completed/Failed."""
from __future__ import annotations

import json
import time
import uuid

from torque.evolus.client import EvolusClient, EvolusError

_MOCK_RESULTS: dict[str, dict] = {}


def execute(c: EvolusClient, input_text: str, parameters: dict, callback_url: str | None = None, mock_output: dict | None = None) -> str:
    wf = c.s.evolus_workflow_id
    path = f"/workflows/{wf}/execute"
    if c.mock or not wf:
        eid = f"mock-exec-{uuid.uuid4().hex[:8]}"
        _MOCK_RESULTS[eid] = {"executionId": eid, "status": "Completed", "output": json.dumps(mock_output or {})}
        c.audit_mock("workflow.execute", path, f"params={parameters}")
        return eid
    form = {
        "input": input_text,
        # TODO: confirm in workspace API reference - the form field name for the flat-JSON parameters object
        # ("parameters" assumed). Never put secrets here: Evolus stores parameters with the run.
        "parameters": json.dumps(parameters),
    }
    if callback_url:
        form.update({"callbackUrl": callback_url, "callbackMethod": "POST", "callbackApiKeyName": "X-Torque-Key"})
        if c.s.callback_key:
            form["callbackApiKey"] = c.s.callback_key
    # multipart/form-data: send fields as (None, value) file tuples so httpx encodes multipart.
    r = c.request("workflow.execute", "POST", path, files={k: (None, v) for k, v in form.items()})
    if r.status_code == 409:
        raise EvolusError("workflow disabled (409): turn the workflow ON in the workspace", 409, r.text)
    data = c.json_or_raise(r, "workflow.execute")
    eid = data.get("executionId") or data.get("id")
    if not eid:
        raise EvolusError(f"workflow.execute: no executionId in {data}")
    return eid


def get_execution(c: EvolusClient, execution_id: str) -> dict:
    wf = c.s.evolus_workflow_id
    path = f"/workflows/{wf}/executions/{execution_id}"
    if execution_id.startswith("mock-exec-"):
        c.audit_mock("workflow.status", path)
        return _MOCK_RESULTS.get(execution_id, {"status": "Failed"})
    r = c.request("workflow.status", "GET", path)
    return c.json_or_raise(r, "workflow.status")


def output_text(execution: dict) -> str:
    """Extract the final output text from an execution/callback payload.
    TODO: confirm in workspace API reference - exact field holding the last step output."""
    for k in ("output", "result", "finalOutput", "response", "text"):
        v = execution.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (dict, list)) and v:
            return json.dumps(v)
    steps = execution.get("steps") or execution.get("stepResults") or []
    if steps:
        last = steps[-1]
        return last.get("output") if isinstance(last.get("output"), str) else json.dumps(last.get("output", last))
    return json.dumps(execution)


def wait(c: EvolusClient, execution_id: str, store=None, timeout_s: float = 90.0, interval_s: float = 3.0) -> dict:
    """Prefer the callback (stored by /callback/evolus); otherwise poll every 3 s (60 req/min limit)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if store is not None:
            cb = store.get_callback(execution_id)
            if cb:
                return {"status": "Completed", **cb}
        ex = get_execution(c, execution_id)
        st = str(ex.get("status", "")).lower()
        if st in ("completed", "succeeded", "success"):
            return ex
        if st in ("failed", "error", "cancelled", "canceled"):
            raise EvolusError(f"workflow execution {execution_id} {ex.get('status')}", None, json.dumps(ex)[:500])
        time.sleep(interval_s)
    raise EvolusError(f"workflow execution {execution_id} timed out after {timeout_s}s")
