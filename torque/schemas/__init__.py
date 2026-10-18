"""JSON schemas from PRD §9 and validation helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def schema(name: str) -> dict:
    return json.loads((_DIR / f"{name}.json").read_text(encoding="utf-8"))


def errors(name: str, obj) -> list[str]:
    v = Draft202012Validator(schema(name))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in sorted(v.iter_errors(obj), key=str)]
