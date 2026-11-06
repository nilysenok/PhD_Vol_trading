"""Поток LGBM волны B (Ж-206): python3 models_v2/s6/lgbm/run_b.py --stage lb0|loss|feat1|feat2|abl|arch   (процессов <= 4; без коммита)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6.lgbm import st_arch, st_lb0, st_run, world  # noqa: E402

STAGES = {"lb0": st_lb0.stage, "loss": st_run.loss, "feat1": st_run.feat1, "feat2": st_run.feat2, "abl": st_run.abl, "arch": st_arch.stage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=tuple(STAGES), required=True)
    a = ap.parse_args()
    W = world.load()
    STAGES[a.stage](W)


if __name__ == "__main__":
    main()
