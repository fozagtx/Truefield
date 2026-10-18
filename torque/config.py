"""Runtime settings, read from environment variables (see .env.example).

Values that still contain a template placeholder such as ``<DROPLET_IP>`` are treated as unset,
so a copied-but-unfilled .env keeps the app in mock mode instead of failing.
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
    # AMD / vLLM
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
    http_timeout_s: float

    @property
    def evolus_enabled(self) -> bool:
        return bool(self.evolus_api_key)

    @property
    def llm_mode(self) -> str:
        """Which backend answers diagnosis calls: evolus | vllm | mock."""
        p = self.llm_provider
        if p in {"evolus", "vllm", "mock"}:
            return p
        if self.evolus_api_key and self.evolus_agent_model:
            return "evolus"
        if self.vllm_base_url:
            return "vllm"
        return "mock"


def get_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(REPO_ROOT / "data")))
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    model_path = Path(_env("MODEL_PATH", str(REPO_ROOT / "models" / "ae-v1.pt")))
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    return Settings(
        vllm_base_url=_env("VLLM_BASE_URL").rstrip("/"),
        vllm_api_key=_env("VLLM_API_KEY"),
        vllm_model=_env("VLLM_MODEL", "google/gemma-4-26B-A4B-it"),
        evolus_api_key=_env("EVOLUS_API_KEY"),
        evolus_app_id=_env("EVOLUS_APP_ID"),
        evolus_agent_id=_env("EVOLUS_AGENT_ID"),
        evolus_agent_model=_env("EVOLUS_AGENT_MODEL"),
        evolus_library_id=_env("EVOLUS_LIBRARY_ID"),
        evolus_workflow_id=_env("EVOLUS_WORKFLOW_ID"),
        evolus_extractor_id=_env("EVOLUS_EXTRACTOR_ID"),
        evolus_extractor_agent_id=_env("EVOLUS_EXTRACTOR_AGENT_ID"),
        evolus_rest_base=_env("EVOLUS_REST_BASE", "https://api.evolus.ai/api/v1").rstrip("/"),
        evolus_chat_base=_env("EVOLUS_CHAT_BASE", "https://api.evolus.ai/v1").rstrip("/"),
        evolus_minutes_tool=_env("EVOLUS_MINUTES_TOOL", "minutes"),
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        public_base_url=_env("PUBLIC_BASE_URL").rstrip("/"),
        callback_key=_env("TORQUE_CALLBACK_KEY"),
        demo_cache=_flag("DEMO_CACHE"),
        demo_cache_record=_flag("DEMO_CACHE_RECORD"),
        cost_per_hour_usd=float(_env("COST_PER_HOUR_USD", "36000") or 36000),
        planned_minutes=int(_env("PLANNED_REPAIR_MINUTES", "90") or 90),
        unplanned_minutes=int(_env("UNPLANNED_BASELINE_MINUTES", "480") or 480),
        data_dir=data_dir,
        model_path=model_path,
        dataset=_env("TORQUE_DATASET", "auto").lower(),
        auto_pipeline=_flag("AUTO_PIPELINE", "1"),
        mock_guardrail_demo=_flag("MOCK_GUARDRAIL_DEMO", "1"),
        http_timeout_s=float(_env("HTTP_TIMEOUT_S", "30") or 30),
    )
