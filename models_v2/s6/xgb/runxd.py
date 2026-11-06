"""Волна D (Ж-206), поток XGB: `python3 -W ignore models_v2/s6/xgb/runxd.py --stage <control|d1|d1sel|d2|r1|r4|tables>`, строго по порядку, по одной. Процессов <= 4.
Прежние каталоги и модули вне d_* только читаются; кэш и строки — `data/regen/xgb_lgbm_full_v1/xgboost/d/`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6 import paths6  # noqa: E402
from models_v2.s6.xgb import d_stages, d_stages2, worldx  # noqa: E402

STAGES = {"control": d_stages.stage_control, "d1": d_stages.stage_d1, "d1sel": d_stages.stage_d1sel, "d2": d_stages2.stage_d2, "r1": d_stages2.stage_r1,
          "r4": d_stages2.stage_r4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=(*STAGES, "tables"))
    a = ap.parse_args()
    if a.stage == "tables":
        return d_stages2.stage_tables()
    paths6.mkdirs()
    return STAGES[a.stage](worldx.load())


if __name__ == "__main__":
    sys.exit(main())
