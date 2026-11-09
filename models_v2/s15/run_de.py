"""Запуск листов Deep Structural Extension (SBER, h = 1). `--stage 0` пишет таблицы DE0, отчёт и CHECKPOINT ключ `0`; `--stage 0 --check` только читает.
python3 -W ignore -m models_v2.s15.run_de --stage 0 [--check]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    from models_v2.s15 import de0_stage as S
    if a.check:
        bad = S.check()
        print("DE0 --check: %s" % ("0 расхождений" if not bad else "расхождений %d: %s" % (len(bad), "; ".join(bad))))
        return 1 if bad else 0
    S.write()
    print("DE0: таблицы, отчёт и CHECKPOINT[0] записаны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
