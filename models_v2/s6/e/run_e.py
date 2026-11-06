"""Волна E Ж-206 + Ж-210: python3 -W ignore models_v2/s6/e/run_e.py --stage comp|ex0|ex1|a|b|c|seeds|e3|e4|e5   (строго по порядку, по одной; процессов <= 4; без коммита)."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

STAGES = {"comp": "st_comp", "ex0": "st_ex0", "ex1": "st_ex1", "a": "st_a", "b": "st_b", "c": "st_c", "seeds": "st_seeds", "e3": "st_e3", "e4": "st_e4", "e5": "st_e5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--only", default="", help="стадия b: список рядов через запятую (кэшируются)")
    ap.add_argument("--reuse-f", action="store_true", help="стадия e4: для B не переобучать, взять F из прежнего HYBRID_SELECTION.csv (только пересборка таблиц)")
    a = ap.parse_args()
    return importlib.import_module("models_v2.s6.e." + STAGES[a.stage]).stage(a)


if __name__ == "__main__":
    sys.exit(main())
