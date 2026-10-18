"""Telemetry sources for replay (FR-1).

Priority: (1) preprocessed demo slice  DATA_DIR/demo/ims_set2_b1.npz  (scripts/make_demo_slice.py)
          (2) raw IMS Set 2 folder       DATA_DIR/ims/2nd_test         (scripts/download_ims.sh)
          (3) synthetic fallback (offline demo), clearly labelled SYNTHETIC.
Force one with TORQUE_DATASET=demo|raw|synthetic.

The housing temperature channel is ALWAYS synthetic (IMS has no temperature): it is derived from
the measured vibration RMS ratio to the healthy baseline (friction heating proxy), plus noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from torque.anomaly.features import FS_HZ
from torque.data import synthetic

log = logging.getLogger("torque.data")


@dataclass
class Dataset:
    signals: np.ndarray  # (n, 20480) float32, bearing 1 channel
    timestamps: list
    source: str  # IMS_SET2_REAL | SYNTHETIC
    label: str
    fs_hz: float = FS_HZ
    temps: np.ndarray = field(default=None)

    @property
    def n(self) -> int:
        return len(self.signals)

    @property
    def end_of_test(self) -> datetime:
        return self.timestamps[-1]

    @property
    def badge(self) -> str:
        if self.source == "IMS_SET2_REAL":
            return "NASA IMS Set 2 (real) + temperature (synthetic)"
        return "SYNTHETIC vibration (offline fallback) + temperature (synthetic)"


def synthetic_temperature(signals: np.ndarray, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = signals.astype(np.float64)
    rms = np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1))
    n0 = max(3, len(rms) // 10)
    rms0 = float(np.median(rms[:n0]))
    ratio = rms / rms0 if rms0 > 0 else np.ones_like(rms)
    temp = 45.0 + 8.0 * np.clip(ratio - 1.0, 0, None) + rng.normal(0, 0.4, len(rms))
    return np.clip(temp, 20.0, 110.0).round(2)


def _parse_ims_name(name: str) -> datetime:
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def load_raw_ims(folder: Path, channel: int = 0, stride: int = 1):
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name[:4].isdigit())
    files = files[::stride]
    sigs = np.empty((len(files), 20480), dtype=np.float32)
    ts = []
    for i, f in enumerate(files):
        arr = np.loadtxt(f, dtype=np.float32)
        sigs[i] = arr[:, channel] if arr.ndim == 2 else arr
        ts.append(_parse_ims_name(f.name))
    return sigs, ts


def load_dataset(data_dir: Path, which: str = "auto") -> Dataset:
    demo = data_dir / "demo" / "ims_set2_b1.npz"
    raw = data_dir / "ims" / "2nd_test"
    ds = None
    if which in {"auto", "demo"} and demo.exists():
        z = np.load(demo, allow_pickle=False)
        ts = [datetime.fromisoformat(s) for s in z["timestamps"].tolist()]
        ds = Dataset(z["signals"].astype(np.float32), ts, "IMS_SET2_REAL", f"NASA IMS Set 2, bearing 1 ({demo.name})")
    elif which in {"auto", "raw"} and raw.exists() and any(raw.iterdir()):
        sigs, ts = load_raw_ims(raw)
        ds = Dataset(sigs, ts, "IMS_SET2_REAL", "NASA IMS Set 2, bearing 1 (raw)")
    if ds is None:
        sigs, ts = synthetic.generate()
        ds = Dataset(sigs, ts, "SYNTHETIC", "Synthetic outer-race degradation (offline fallback)")
    ds.temps = synthetic_temperature(ds.signals)
    log.info("dataset loaded: %s, %d snapshots", ds.label, ds.n)
    return ds
