"""Дописывает все строки ledger В1 в docs/otbor/reestr_otbora_A_v1.csv (только дополнение, без дублей).
Запуск: python3 scripts/reestr_v1_iz_ledger.py --it 1 2 3 4 [--sk S1 S2 ...]."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastbt import reestr_otbora_v1 as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, nargs="+", required=True)
    ap.add_argument("--sk", nargs="+", default=[f"S{k}" for k in range(1, 7)])
    ap.add_argument("--period", default="train4", choices=sorted(R.PERIODY))
    a = ap.parse_args(argv)
    for it in a.it:
        for sk in a.sk:
            print(f"it{it} {sk}: добавлено {R.iz_ledger(a.period, it, sk)}")


if __name__ == "__main__":
    main()
