"""Merge bench outputs (ae_results.json, vllm_c*.json, e2e_results.json) into bench/results.json for the /bench page."""
import json
from pathlib import Path

D = Path(__file__).parent
KEYS = ["output_throughput", "total_token_throughput", "mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "completed", "max_concurrency"]


def load(name):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == "__main__":
    res = {"autoencoder": load("ae_results.json"), "end_to_end": load("e2e_results.json"), "vllm": {}}
    for c in (1, 32):
        v = load(f"vllm_c{c}.json")
        res["vllm"][f"concurrency_{c}"] = {k: v.get(k) for k in KEYS} if v else None
    res["note"] = "Numbers produced by the scripts in bench/. Fill GPU/vLLM rows by running them on the AMD MI300X droplet."
    (D / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
