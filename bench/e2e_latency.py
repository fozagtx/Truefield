"""End-to-end latency against a running API: alert -> diagnosis and alert -> work order, p50/p95 over N runs.
  python bench/e2e_latency.py --api http://localhost:8080 --runs 10
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).with_name("e2e_results.json")))
    a = ap.parse_args()
    dx_s, wo_s = [], []
    for i in range(a.runs):
        requests.post(f"{a.api}/reset", timeout=30)
        requests.post(f"{a.api}/replay/start", params={"speed": 1000, "start_frac": 0.6}, timeout=30)
        ev = None
        for _ in range(600):
            evs = requests.get(f"{a.api}/events", timeout=30).json()
            ev = next((e for e in evs if e["rules"]["severity"] in ("warning", "alarm", "critical")), None)
            if ev:
                break
            time.sleep(0.1)
        requests.post(f"{a.api}/replay/stop", timeout=30)
        t0 = time.perf_counter()
        requests.post(f"{a.api}/events/{ev['event_id']}/diagnose", params={"force": True}, timeout=180)
        t1 = time.perf_counter()
        requests.post(f"{a.api}/events/{ev['event_id']}/workorder", timeout=300)
        t2 = time.perf_counter()
        dx_s.append(t1 - t0); wo_s.append(t2 - t0)
        print(f"run {i + 1}: diagnosis {t1 - t0:.2f}s, work order {t2 - t0:.2f}s")
    health = requests.get(f"{a.api}/health", timeout=30).json()

    def q(v, p):
        return round(statistics.quantiles(v, n=100)[p - 1], 3) if len(v) > 1 else round(v[0], 3)
    out = {"runs": a.runs, "modes": health["modes"], "alert_to_diagnosis_s": {"p50": q(dx_s, 50), "p95": q(dx_s, 95)},
           "alert_to_work_order_s": {"p50": q(wo_s, 50), "p95": q(wo_s, 95)}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
