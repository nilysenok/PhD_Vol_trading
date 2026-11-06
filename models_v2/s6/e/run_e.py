"""Волна E Ж-206: python3 -W ignore models_v2/s6/e/run_e.py --stage comp|e1|resid|e2|seeds|e3|e4|e5   (строго по порядку, по одной; процессов <= 4; без коммита)."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

STAGES = {"comp": "st_comp", "e1": "st_e1", "resid": "resid", "e2": "st_e2", "seeds": "st_seeds", "e3": "st_e3", "e4": "st_e4", "e5": "st_e5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    a = ap.parse_args()
    return importlib.import_module("models_v2.s6.e." + STAGES[a.stage]).stage(a)


if __name__ == "__main__":
    sys.exit(main())
