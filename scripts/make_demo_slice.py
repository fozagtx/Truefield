"""Preprocess IMS Set 2 (bearing 1 channel) into a compact demo slice: DATA_DIR/demo/ims_set2_b1.npz.

  python scripts/make_demo_slice.py --stride 2      # 492 snapshots, float16, ~20 MB (not committed)
The app picks it up automatically (TRUEFIELD_DATASET=auto). Retrain the AE after changing it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from truefield.data.dataset import load_raw_ims  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ims/2nd_test")
    ap.add_argument("--out", default="data/demo/ims_set2_b1.npz")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth snapshot (the last one is always kept)")
    ap.add_argument("--channel", type=int, default=0, help="0 = bearing 1 (outer-race failure)")
    a = ap.parse_args()
    sigs, ts = load_raw_ims(Path(a.src), channel=a.channel)
    idx = list(range(0, len(sigs), a.stride))
    if idx[-1] != len(sigs) - 1:
        idx.append(len(sigs) - 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, signals=sigs[idx].astype(np.float16), timestamps=np.array([ts[i].isoformat() for i in idx]))
    print(f"wrote {out} with {len(idx)} snapshots ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
