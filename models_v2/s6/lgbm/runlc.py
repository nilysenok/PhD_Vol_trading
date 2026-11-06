"""Поток LGBM волны C (Ж-206): python3 models_v2/s6/lgbm/runlc.py --stage l0|l1|l2|nbr|leaves   (процессов <= 4; без коммита)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models_v2.s6.lgbm import c_l0, world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("l0", "l1", "l2", "nbr", "leaves"), required=True)
    a = ap.parse_args()
    W = world.load()
    if a.stage == "l0":
        c_l0.stage(W)
    elif a.stage == "l1":
        from models_v2.s6.lgbm import c_l1
        c_l1.stage(W)
    elif a.stage == "l2":
        from models_v2.s6.lgbm import c_l2
        c_l2.stage(W)
    elif a.stage == "nbr":
        from models_v2.s6.lgbm import c_nbr
        c_nbr.stage(W)
    else:
        from models_v2.s6.lgbm import c_leaves
        c_leaves.stage(W)


if __name__ == "__main__":
    main()
