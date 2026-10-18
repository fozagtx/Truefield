"""Deterministic condition indicators for one vibration snapshot (FR-2).

For each 20,480-point snapshot sampled at 20 kHz we compute:
  rms, peak, crest factor, kurtosis (Pearson, normal = 3),
  envelope-spectrum energy within +/-3 Hz of BPFO, BPFI, BSF and FTF.

The envelope is the magnitude of the analytic signal (Hilbert transform) of the signal
band-passed around the structural resonance band (2-8 kHz), which is where bearing impacts ring.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

FS_HZ = 20000.0
N_POINTS = 20480
# IMS rig (Rexnord ZA-2115, shaft 2000 rpm = 33.3 Hz) characteristic frequencies, per PRD Appendix A.
FAULT_FREQS_HZ = {"bpfo": 236.0, "bpfi": 297.0, "bsf": 278.0, "ftf": 15.0}
BAND_HALF_WIDTH_HZ = 3.0
ENVELOPE_BAND_HZ = (2000.0, 8000.0)
N_SPECTRAL_BANDS = 32
FEATURE_NAMES = ["rms", "peak", "crest", "kurtosis", "env_bpfo", "env_bpfi", "env_bsf", "env_ftf"]


@lru_cache(maxsize=8)
def _bandpass_sos(fs: float, lo: float, hi: float):
    return butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")


def envelope_spectrum(x: np.ndarray, fs: float = FS_HZ, band=ENVELOPE_BAND_HZ):
    """Return (freqs, power) of the envelope spectrum."""
    x = np.asarray(x, dtype=np.float64)
    y = sosfiltfilt(_bandpass_sos(float(fs), float(band[0]), float(band[1])), x - x.mean())
    env = np.abs(hilbert(y))
    env = env - env.mean()
    power = np.abs(np.fft.rfft(env)) ** 2 / len(env)
    freqs = np.fft.rfftfreq(len(env), 1.0 / fs)
    return freqs, power


def band_energy(freqs: np.ndarray, power: np.ndarray, f0: float, half_width: float = BAND_HALF_WIDTH_HZ) -> float:
    m = (freqs >= f0 - half_width) & (freqs <= f0 + half_width)
    return float(power[m].sum())


def extract_features(x: np.ndarray, fs: float = FS_HZ, fault_freqs: dict | None = None) -> dict:
    ff = fault_freqs or FAULT_FREQS_HZ
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(xc**2)))
    peak = float(np.max(np.abs(xc)))
    crest = float(peak / rms) if rms > 0 else 0.0
    var = np.mean(xc**2)
    kurt = float(np.mean(xc**4) / var**2) if var > 0 else 0.0
    freqs, power = envelope_spectrum(x, fs)
    feats = {"rms": rms, "peak": peak, "crest": crest, "kurtosis": kurt}
    for k in ("bpfo", "bpfi", "bsf", "ftf"):
        feats[f"env_{k}"] = band_energy(freqs, power, ff[k])
    return feats


def spectral_bands(x: np.ndarray, fs: float = FS_HZ, n_bands: int = N_SPECTRAL_BANDS) -> np.ndarray:
    """Log-FFT band energies (autoencoder input): mean |X| in n linear bands up to Nyquist, log10."""
    x = np.asarray(x, dtype=np.float64)
    mag = np.abs(np.fft.rfft(x - x.mean()))[1:]
    usable = (len(mag) // n_bands) * n_bands
    bands = mag[:usable].reshape(n_bands, -1).mean(axis=1)
    return np.log10(bands + 1e-9).astype(np.float32)


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)
