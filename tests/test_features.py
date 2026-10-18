import time

import numpy as np

from truefield.anomaly.features import FS_HZ, N_POINTS, extract_features, spectral_bands
from truefield.data.synthetic import generate


def test_sine_known_values():
    t = np.arange(N_POINTS) / FS_HZ
    x = 2.0 * np.sin(2 * np.pi * 1000 * t)
    f = extract_features(x)
    assert abs(f["rms"] - 2 / np.sqrt(2)) < 1e-3
    assert abs(f["peak"] - 2.0) < 1e-3
    assert abs(f["crest"] - np.sqrt(2)) < 1e-3
    assert abs(f["kurtosis"] - 1.5) < 1e-3  # sine kurtosis (Pearson) = 1.5


def test_deterministic_tolerance():
    sig, _ = generate(n=3, seed=1)
    a, b = extract_features(sig[2]), extract_features(sig[2].copy())
    for k in a:
        assert abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k]))


def test_bpfo_band_detects_outer_race():
    sig, _ = generate(n=50, seed=3)
    healthy, faulty = extract_features(sig[0]), extract_features(sig[-1])
    assert faulty["env_bpfo"] > 50 * healthy["env_bpfo"]
    assert faulty["env_bpfo"] > max(faulty["env_bpfi"], faulty["env_bsf"], faulty["env_ftf"])
    assert faulty["kurtosis"] > healthy["kurtosis"]


def test_speed_under_50ms():
    sig, _ = generate(n=2, seed=4)
    extract_features(sig[0])
    t0 = time.perf_counter()
    for _ in range(5):
        extract_features(sig[1])
    assert (time.perf_counter() - t0) / 5 < 0.05


def test_spectral_bands_shape():
    sig, _ = generate(n=1)
    assert spectral_bands(sig[0]).shape == (32,)
