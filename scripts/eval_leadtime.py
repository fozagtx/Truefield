"""Offline lead-time evaluation: replay the whole record through AE + rules and compare with a naive 2x RMS alarm."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torque.config import get_settings  # noqa: E402
from torque.kpi import evaluate_lead_time  # noqa: E402
from torque.pipeline import Torque  # noqa: E402

if __name__ == "__main__":
    t = Torque(get_settings())
    res = evaluate_lead_time(t)
    print(json.dumps(res, indent=2))
    print(f"\nTorque alarmed {res['torque_first_alert_lead_h']} h before end of test vs {res['naive_2x_rms_lead_h']} h for a 2x-RMS threshold ({t.ds.badge}).")
