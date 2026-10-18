"""Train the autoencoder on the healthy window (first 40%) and save models/ae-v1.pt.

  python -m truefield.anomaly.train --out models/ae-v1.pt                 # uses DATA_DIR dataset (or synthetic)
  python -m truefield.anomaly.train --data data/ims/2nd_test --out models/ae-v1.pt   # raw IMS on the droplet
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from truefield.anomaly.features import FEATURE_NAMES, extract_features, feature_vector, spectral_bands
from truefield.anomaly.model import AutoEncoder, get_device
from truefield.data.dataset import Dataset, load_dataset, synthetic_temperature, load_raw_ims

log = logging.getLogger("truefield.anomaly")
HEALTHY_FRACTION = 0.4
THRESHOLD_PERCENTILE = 99.5


def compute_matrices(ds: Dataset):
    bands = np.stack([spectral_bands(s, ds.fs_hz) for s in ds.signals])
    feats = [extract_features(s, ds.fs_hz) for s in ds.signals]
    return bands, feats


def train(ds: Dataset, out: Path, epochs: int = 100, device_pref: str = "auto", seed: int = 0, precomputed=None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, device_name = get_device(device_pref)
    t0 = time.perf_counter()
    bands, feats = precomputed or compute_matrices(ds)
    feat_secs = time.perf_counter() - t0
    n_h = max(8, int(ds.n * HEALTHY_FRACTION))
    hb = bands[:n_h]
    mu, sd = hb.mean(axis=0), hb.std(axis=0) + 1e-6
    xh = torch.tensor((hb - mu) / sd, dtype=torch.float32, device=device)
    model = AutoEncoder(n_in=bands.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    t1 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xh), device=device)
        for i in range(0, len(xh), 32):
            b = xh[perm[i : i + 32]]
            opt.zero_grad()
            loss = lossf(model(b), b)
            loss.backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_secs = time.perf_counter() - t1
    model.eval()
    with torch.no_grad():
        err_h = ((model(xh) - xh) ** 2).mean(dim=1).cpu().numpy()
    threshold = float(np.percentile(err_h, THRESHOLD_PERCENTILE))
    fv = np.stack([feature_vector(f) for f in feats[:n_h]])
    temps = ds.temps if ds.temps is not None else synthetic_temperature(ds.signals)
    meta = {
        "name": "ae-v1",
        "trained_on": ds.label,
        "source": ds.source,
        "n_snapshots": int(ds.n),
        "n_healthy": int(n_h),
        "epochs": int(epochs),
        "device": device_name,
        "feature_seconds": round(feat_secs, 3),
        "train_seconds": round(train_secs, 3),
        "threshold": threshold,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "band_mu": mu.tolist(),
        "band_sd": sd.tolist(),
        "feature_names": FEATURE_NAMES,
        "feature_mu": fv.mean(axis=0).tolist(),
        "feature_sd": fv.std(axis=0).tolist(),
        "temp_baseline_c": float(np.mean(temps[:n_h])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "meta": meta}, out)
    log.info("saved %s on %s: threshold=%.4f train=%.2fs", out, device_name, threshold, train_secs)
    return meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="raw IMS folder (e.g. data/ims/2nd_test); default: DATA_DIR dataset")
    ap.add_argument("--out", default="models/ae-v1.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    args = ap.parse_args()
    from truefield.config import get_settings

    if args.data:
        sigs, ts = load_raw_ims(Path(args.data))
        ds = Dataset(sigs, ts, "IMS_SET2_REAL", f"NASA IMS Set 2, bearing 1 ({args.data})")
        ds.temps = synthetic_temperature(sigs)
    else:
        s = get_settings()
        ds = load_dataset(s.data_dir, s.dataset)
    meta = train(ds, Path(args.out), epochs=args.epochs, device_pref=args.device)
    print({k: v for k, v in meta.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
