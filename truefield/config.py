"""Runtime settings, read from environment variables (see .env.example).

Values that still contain a template placeholder such as ``<DROPLET_IP>`` are treated as unset,
so a copied-but-unfilled .env counts as not configured (the API then fails fast at startup).
Mock mode is never automatic: only MOCK=1 or LLM_PROVIDER=mock enables it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    v = (v or "").strip()
    if "<" in v and ">" in v:  # unfilled template placeholder
        return ""
    return v


def _flag(name: str, default: str = "0") -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Primary LLM: OpenAI-compatible inference API (DigitalOcean serverless inference by default)
    model_access_key: str
    model_base_url: str
    model_name: str
    # Optional alternative: our own vLLM (Gemma on AMD)
    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    # Evolus
    evolus_api_key: str
    evolus_app_id: str
    evolus_agent_id: str
    evolus_agent_model: str
    evolus_library_id: str
    evolus_workflow_id: str
    evolus_extractor_id: str
    evolus_extractor_agent_id: str
    evolus_rest_base: str
    evolus_chat_base: str
    evolus_minutes_tool: str
    # App
    llm_provider: str
    public_base_url: str
    callback_key: str
    demo_cache: bool
    demo_cache_record: bool
    cost_per_hour_usd: float
    planned_minutes: int
    unplanned_minutes: int
    data_dir: Path
    model_path: Path
    dataset: str
    auto_pipeline: bool
    mock_guardrail_demo: bool
    mock: bool  # explicit only: MOCK=1 or LLM_PROVIDER=mock
    http_timeout_s: float

    @property
    def evolus_enabled(self) -> bool:
        return bool(self.evolus_api_key)

    @property
    def llm_mode(self) -> str:
        """Which backend answers diagnosis calls: do | evolus | vllm | mock | none.
        mock only when explicitly enabled. auto: MODEL_ACCESS_KEY -> do (primary); else Evolus agent; else vLLM;
        else "none" (the API refuses to start)."""
        if self.mock:
            return "mock"
        p = self.llm_provider
        if p in {"do", "evolus", "vllm"}:
            return p
        if self.model_access_key:
            return "do"
        if self.evolus_api_key and self.evolus_agent_model:
            return "evolus"
        if self.vllm_base_url:
            return "vllm"
        return "none"


def get_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(REPO_ROOT / "data")))
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    model_path = Path(_env("MODEL_PATH", str(REPO_ROOT / "models" / "ae-v1.pt")))
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    app_id, agent_id = _env("EVOLUS_APP_ID"), _env("EVOLUS_AGENT_ID")
    return Settings(
        model_access_key=_env("MODEL_ACCESS_KEY"),
        model_base_url=_env("MODEL_BASE_URL", "https://inference.do-ai.run/v1").rstrip("/"),
        model_name=_env("MODEL_NAME", "gpt-4o"),
        vllm_base_url=_env("VLLM_BASE_URL").rstrip("/"),
        vllm_api_key=_env("VLLM_API_KEY"),
        vllm_model=_env("VLLM_MODEL", "google/gemma-4-26B-A4B-it"),
        evolus_api_key=_env("EVOLUS_API_KEY"),
        evolus_app_id=app_id,
        evolus_agent_id=agent_id,
        # derived: "<applicationId>:<agentId>" (override with EVOLUS_AGENT_MODEL if GET /v1/models differs)
        evolus_agent_model=_env("EVOLUS_AGENT_MODEL") or (f"{app_id}:{agent_id}" if app_id and agent_id else ""),
        evolus_library_id=_env("EVOLUS_LIBRARY_ID"),
        evolus_workflow_id=_env("EVOLUS_WORKFLOW_ID"),
        evolus_extractor_id=_env("EVOLUS_EXTRACTOR_ID"),
        evolus_extractor_agent_id=_env("EVOLUS_EXTRACTOR_AGENT_ID") or agent_id,
        evolus_rest_base=_env("EVOLUS_REST_BASE", "https://api.evolus.ai/api/v1").rstrip("/"),
        evolus_chat_base=_env("EVOLUS_CHAT_BASE", "https://api.evolus.ai/v1").rstrip("/"),
        evolus_minutes_tool=_env("EVOLUS_MINUTES_TOOL", "minutes"),
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        public_base_url=_env("PUBLIC_BASE_URL").rstrip("/"),
        callback_key=_env("TRUEFIELD_CALLBACK_KEY"),
        demo_cache=_flag("DEMO_CACHE"),
        demo_cache_record=_flag("DEMO_CACHE_RECORD"),
        cost_per_hour_usd=float(_env("COST_PER_HOUR_USD", "36000") or 36000),
        planned_minutes=int(_env("PLANNED_REPAIR_MINUTES", "90") or 90),
        unplanned_minutes=int(_env("UNPLANNED_BASELINE_MINUTES", "480") or 480),
        data_dir=data_dir,
        model_path=model_path,
        dataset=_env("TRUEFIELD_DATASET", "auto").lower(),
        auto_pipeline=_flag("AUTO_PIPELINE", "1"),
        mock_guardrail_demo=_flag("MOCK_GUARDRAIL_DEMO", "1"),
        mock=_flag("MOCK") or _env("LLM_PROVIDER", "auto").lower() == "mock",
        http_timeout_s=float(_env("HTTP_TIMEOUT_S", "30") or 30),
    )
