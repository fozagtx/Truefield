"""PyTorch autoencoder on log-FFT bands (FR-3). ROCm-compatible: ROCm exposes AMD GPUs via torch.cuda."""
from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger("truefield.anomaly")


class AutoEncoder(nn.Module):
    def __init__(self, n_in: int = 32, hidden: int = 16, latent: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_device(prefer: str = "auto") -> tuple[torch.device, str]:
    """Return (device, human-readable name). cuda covers both NVIDIA CUDA and AMD ROCm/HIP builds."""
    if prefer != "cpu" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        desc = f"{name} ({backend})"
        log.info("torch device: %s", desc)
        return torch.device("cuda"), desc
    log.info("torch device: CPU (no ROCm/CUDA GPU visible)")
    return torch.device("cpu"), "CPU"
