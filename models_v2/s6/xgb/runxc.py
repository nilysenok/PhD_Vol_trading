"""Волна C (Ж-206), поток XGB: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage <control|x1|x2|x3|seeds|tables>`, строго по порядку, по одной. Процессов <= 4.
Прежние каталоги и модули s6/xgb вне c_* только читаются; кэш и строки — `data/regen/xgb_lgbm_full_v1/xgboost/c/`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6 import paths6  # noqa: E402
from models_v2.s6.xgb import c_rows, c_seeds, c_stages, c_tabs, c_x2, worldx  # noqa: E402

STAGES = {"control": c_stages.stage_control, "x1": c_stages.stage_x1, "x2": c_x2.stage_x2, "x3": c_stages.stage_x3, "seeds": c_seeds.stage_seeds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=(*STAGES, "tables"))
    a = ap.parse_args()
    if a.stage == "tables":
        return c_tabs.stage_tables(c_rows.load_rows())
    paths6.mkdirs()
    return STAGES[a.stage](worldx.load())


if __name__ == "__main__":
    sys.exit(main())
