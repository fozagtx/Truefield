import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests always run offline in mock mode on the synthetic dataset.
for k in ["EVOLUS_API_KEY", "VLLM_BASE_URL", "EVOLUS_AGENT_MODEL", "DEMO_CACHE", "DEMO_CACHE_RECORD", "PUBLIC_BASE_URL"]:
    os.environ.pop(k, None)
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TORQUE_DATASET"] = "synthetic"
