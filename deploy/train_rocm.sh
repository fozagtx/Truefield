#!/usr/bin/env bash
# Train the autoencoder on the MI300X with ROCm PyTorch (logs torch.cuda.get_device_name(0) -> "AMD Instinct MI300X").
# Uses data/demo/ims_set2_b1.npz if present (scripts/make_demo_slice.py), else pass --data data/ims/2nd_test.
set -euo pipefail
docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$PWD":/work -w /work rocm/pytorch:latest \
  bash -c "pip install -q fastapi httpx jsonschema scipy pyyaml python-multipart && \
           python -c 'import torch; print(\"cuda available:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\", \"hip\", torch.version.hip)' && \
           python -m torque.anomaly.train --out models/ae-v1.pt ${*:-} && \
           python bench/ae_bench.py --epochs 50"
