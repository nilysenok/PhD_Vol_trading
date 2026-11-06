"""Поток LGBM волны D (Ж-206): python3 models_v2/s6/lgbm/runld.py --stage win|pick|freq|r1|r4|report   (процессов <= 4; без коммита)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6.lgbm import d_r, d_report, d_run, world  # noqa: E402

STAGES = {"win": d_run.stage_win, "pick": d_run.stage_pick, "freq": d_run.stage_freq, "r1": d_r.stage_r1, "r4": d_r.stage_r4, "report": d_report.stage_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=tuple(STAGES), required=True)
    STAGES[ap.parse_args().stage](world.load())


if __name__ == "__main__":
    main()
