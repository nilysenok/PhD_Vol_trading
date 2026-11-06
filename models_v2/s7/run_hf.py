"""Ж-211 SBER h=1: запуск стадий по одной. `python3 -W ignore models_v2/s7/run_hf.py --stage reg|diag|bound|inter|weights|feat|regime|final`."""
from __future__ import annotations

import argparse
import importlib

STAGES = {"reg": "hf_stage_reg", "diag": "hf_diag", "bound": "hf_bound", "inter": "hf_inter", "weights": "hf_wstage",
          "feat": "hf_feat", "regime": "hf_regstage", "final": "hf_final"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    a = ap.parse_args()
    return importlib.import_module("models_v2.s7." + STAGES[a.stage]).stage(a)


if __name__ == "__main__":
    raise SystemExit(main())
