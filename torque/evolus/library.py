"""Knowledge library (verified): POST /memories/{libraryId}/documents?language=en (multipart file),
POST /memories/{libraryId}/documents/import-url, GET /memories/{libraryId}/documents."""
from __future__ import annotations

from torque.evolus.client import EvolusClient


def upload_document(c: EvolusClient, name: str, content: bytes, mime: str = "text/markdown") -> dict:
    lib = c.s.evolus_library_id
    path = f"/memories/{lib}/documents?language=en"
    if c.mock or not lib:
        c.audit_mock("library.upload", path, name)
        return {"id": f"mock-doc-{name}", "name": name, "status": "mock"}
    r = c.request("library.upload", "POST", path, files={"file": (name, content, mime)})
    return c.json_or_raise(r, "library.upload")


def import_url(c: EvolusClient, url: str) -> dict:
    lib = c.s.evolus_library_id
    path = f"/memories/{lib}/documents/import-url"
    if c.mock or not lib:
        c.audit_mock("library.import_url", path, url)
        return {"id": "mock-url-doc", "url": url, "status": "mock"}
    # TODO: confirm in workspace API reference - body field name for the URL ("url") and language param.
    r = c.request("library.import_url", "POST", path + "?language=en", json={"url": url})
    return c.json_or_raise(r, "library.import_url")


def list_documents(c: EvolusClient) -> list:
    lib = c.s.evolus_library_id
    path = f"/memories/{lib}/documents"
    if c.mock or not lib:
        c.audit_mock("library.list", path)
        return []
    r = c.request("library.list", "GET", path)
    data = c.json_or_raise(r, "library.list")
    # TODO: confirm in workspace API reference - list envelope (bare list vs {"items": [...]}) and indexing status field.
    return data if isinstance(data, list) else data.get("items") or data.get("documents") or data.get("data") or []
