#!/usr/bin/env python3
"""Д17.1: шаг 10 (обучение <= 2024, ES по val 2025, прогноз 01-02.2026) для прошедшего кандидата — final2_step10.py (ветка final3, без изменений)
с обёрткой step_data из d17_1.py (офсет ln основы + s²/2, клип метки gamma; основа HAR-X — D17_BASE=harx, файл D17/harx_base_h{h}.parquet
с шагом 10 — d17_1_harx_base.py --steps 10). Метрики не считаются.
D17_BASE=harx L2_FEAT=<data/regen>/features_pit_v3.parquet python3 scripts/analysis/d17_1_step10.py --kind opt --boost D17_1_h5_G5Xopt --h 5 \
    --cfg '{"obj":"gamma","v9":true,"b01":true,"harf":true}' --blocks d7"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d17_1  # noqa: F401  (подменяет l2_lib.step_data)
import final2_step10 as S10

if __name__ == '__main__':
    if d17_1.BASE_KIND == 'harx':
        d17_1.harx_base(int(sys.argv[sys.argv.index('--h') + 1]), 10)
    S10.main()
