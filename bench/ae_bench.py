"""Autoencoder benchmark: training time (N epochs) and inference throughput, GPU (ROCm/CUDA) vs CPU.
  python bench/ae_bench.py --epochs 50 --out bench/ae_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torque.anomaly.model import AutoEncoder, get_device  # noqa: E402


def run(device: torch.device, x: np.ndarray, epochs: int, batch_infer: int = 1) -> dict:
    torch.manual_seed(0)
    X = torch.tensor(x, dtype=torch.float32, device=device)
    m = AutoEncoder(n_in=x.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for i in range(0, len(X), 32):
            b = X[i:i + 32]
            opt.zero_grad(); loss = ((m(b) - b) ** 2).mean(); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    m.eval()
    res = {"train_seconds": round(train_s, 3)}
    with torch.no_grad():
        for bs in (1, 1024):
            xb = X[:bs] if len(X) >= bs else X.repeat(bs // len(X) + 1, 1)[:bs]
            for _ in range(10):
                m(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            n = 200
            t1 = time.perf_counter()
            for _ in range(n):
                m(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t1) / n
            res[f"latency_ms_batch{bs}"] = round(dt * 1000, 4)
            res[f"snapshots_per_s_batch{bs}"] = round(bs / dt, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n", type=int, default=984, help="snapshots (IMS Set 2 has 984)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("ae_results.json")))
    a = ap.parse_args()
    x = np.random.default_rng(0).standard_normal((a.n, 32)).astype(np.float32)  # standardized log-FFT bands
    out = {"epochs": a.epochs, "n_snapshots": a.n, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
    out["cpu"] = {"device": "CPU", **run(torch.device("cpu"), x, a.epochs)}
    dev, name = get_device()
    if dev.type == "cuda":
        out["gpu"] = {"device": name, **run(dev, x, a.epochs)}
    else:
        out["gpu"] = {"device": None, "note": "no ROCm/CUDA GPU visible; run on the MI300X droplet in rocm/pytorch"}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
