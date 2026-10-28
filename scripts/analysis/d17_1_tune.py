#!/usr/bin/env python3
"""Д17.1: optuna TPE-30 (seed 42) для gamma с офсетом — tune_L2_optuna.py без изменений, с обёрткой step_data из d17_1.py
(офсет ln HAR-J + s²/2 — b01, клип метки gamma). Критерий испытания в tune_L2_optuna — QLIKE exp(прогноза) на val-годе, без поправки.
python3 scripts/analysis/d17_1_tune.py --h 22 --cfg '{"obj":"gamma","v9":true,"b01":true,"linear_lambda":1}' --tag D17_1_h22_G22opt --workers 4"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d17_1  # noqa: F401  (подменяет l2_lib.step_data)
import tune_L2_optuna as T

if __name__ == '__main__':
    if d17_1.BASE_KIND == 'harx':  # основа HAR-X читается один раз до fork
        d17_1.harx_base(int(sys.argv[sys.argv.index('--h') + 1]), 1)
    T.main()
