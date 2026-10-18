"""Per-snapshot scoring: features + z-scores + autoencoder reconstruction error."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from torque.anomaly.features import FEATURE_NAMES, extract_features, spectral_bands
from torque.anomaly.model import AutoEncoder, get_device
from torque.data.dataset import Dataset

log = logging.getLogger("torque.anomaly")


class Scorer:
    def __init__(self, ds: Dataset, model_path: Path, device_pref: str = "auto"):
        self.ds = ds
        self.model_path = model_path
        self._cache: dict[int, dict] = {}
        self._lock = threading.Lock()
        ckpt = None
        if model_path.exists():
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if ckpt["meta"].get("source") != ds.source or ckpt["meta"].get("n_snapshots") != ds.n:
                log.warning("model %s was trained on %s; retraining for %s", model_path, ckpt["meta"].get("trained_on"), ds.label)
                ckpt = None
        if ckpt is None:
            from torque.anomaly.train import train

            log.info("no matching model at %s, training now on %s", model_path, ds.label)
            train(ds, model_path, epochs=100, device_pref=device_pref)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.meta = ckpt["meta"]
        self.device, self.device_name = get_device(device_pref)
        self.model = AutoEncoder(n_in=len(self.meta["band_mu"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.mu = np.array(self.meta["band_mu"], dtype=np.float32)
        self.sd = np.array(self.meta["band_sd"], dtype=np.float32)
        self.fmu = np.array(self.meta["feature_mu"])
        # sigma floor: 5% of the mean, so a very quiet healthy window does not create absurd z-scores
        self.fsd = np.maximum(np.array(self.meta["feature_sd"]), 0.05 * np.abs(self.fmu) + 1e-12)
        self.threshold = float(self.meta["threshold"])
        self.temp_baseline = float(self.meta.get("temp_baseline_c", 45.0))
        self.last_infer_ms = None

    def ae_score(self, signal: np.ndarray) -> float:
        b = (spectral_bands(signal, self.ds.fs_hz) - self.mu) / self.sd
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.tensor(b[None, :], dtype=torch.float32, device=self.device)
            err = float(((self.model(x) - x) ** 2).mean().item())
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        return err

    def score_index(self, i: int) -> dict:
        with self._lock:
            if i in self._cache:
                return self._cache[i]
        sig = self.ds.signals[i]
        feats = extract_features(sig, self.ds.fs_hz)
        z = {k: float((feats[k] - self.fmu[j]) / self.fsd[j]) for j, k in enumerate(FEATURE_NAMES)}
        score = self.ae_score(sig)
        rec = {
            "idx": i,
            "ts": self.ds.timestamps[i].isoformat(),
            "features": feats,
            "z": z,
            "score": score,
            "threshold": self.threshold,
            "above": score > self.threshold,
            "temp_c": float(self.ds.temps[i]),
        }
        with self._lock:
            self._cache[i] = rec
        return rec
