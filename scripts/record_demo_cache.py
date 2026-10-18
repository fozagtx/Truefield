"""Record LLM responses for DEMO_CACHE=1 backup runs: runs one replay -> diagnosis -> WO with DEMO_CACHE_RECORD=1.
Run against your live endpoints (fill .env first). Responses land in DATA_DIR/demo_cache/*.json (git-ignored)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DEMO_CACHE_RECORD"] = "1"
os.environ["DEMO_CACHE"] = "0"
os.environ["AUTO_PIPELINE"] = "0"
from truefield.config import get_settings  # noqa: E402
from truefield.pipeline import Truefield  # noqa: E402

if __name__ == "__main__":
    t = Truefield(get_settings())
    t.store.reset()
    t.start_replay(speed=1000, start=int(0.5 * t.ds.n))
    while t.replay.running and not any(e["rules"]["severity"] in ("alarm", "critical") for e in t.store.list_events()):
        time.sleep(0.2)
    t.stop_replay()
    ev = next(e for e in t.store.list_events() if e["rules"]["severity"] in ("alarm", "critical"))
    dx = t.run_diagnosis(ev["event_id"], force=True)
    wo = t.draft_work_order(ev["event_id"])
    print("recorded diagnosis via", dx["_meta"]["provider"], "and", wo["wo_id"], "drafted by", wo["drafted_by"])
