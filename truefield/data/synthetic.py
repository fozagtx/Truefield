"""Tiny synthetic run-to-failure bearing stream, used ONLY as an offline fallback.

It mimics the IMS Set 2 shape (healthy, then a growing outer-race defect that rings the housing
resonance at the BPFO rate). Every record built from it carries source="SYNTHETIC".
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from truefield.anomaly.features import FAULT_FREQS_HZ, FS_HZ, N_POINTS

IMS_SET2_START = datetime(2004, 2, 12, 10, 32, 39)


def generate(n: int = 300, fs: float = FS_HZ, n_points: int = N_POINTS, seed: int = 7, onset: float = 0.6):
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) / fs
    kt = np.arange(int(0.004 * fs)) / fs
    kernel = np.exp(-kt * 1500.0) * np.sin(2 * np.pi * 4200.0 * kt)
    period = fs / FAULT_FREQS_HZ["bpfo"]
    signals = np.empty((n, n_points), dtype=np.float32)
    for i in range(n):
        p = i / max(n - 1, 1)
        sev = 0.0 if p < onset else ((p - onset) / (1.0 - onset)) ** 2
        x = 0.06 * rng.standard_normal(n_points) + 0.02 * np.sin(2 * np.pi * 33.3 * t + rng.uniform(0, 2 * np.pi))
        if sev > 0:
            starts = np.arange(rng.uniform(0, period), n_points, period)
            idx = np.clip((starts + rng.normal(0, 0.5, len(starts))).astype(int), 0, n_points - 1)
            imp = np.zeros(n_points)
            imp[idx] = sev * 1.5 * (1 + 0.2 * rng.standard_normal(len(idx)))
            x = x + np.convolve(imp, kernel)[:n_points] + sev * 0.08 * rng.standard_normal(n_points)
        signals[i] = x.astype(np.float32)
    timestamps = [IMS_SET2_START + timedelta(minutes=10 * i) for i in range(n)]
    return signals, timestamps
