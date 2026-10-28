#!/usr/bin/env python3
"""etap0: Шарп МСП A/B/C/D (ALL6 x EW, комиссия 0,05%), число дней, бутстрап, CE.

Шарп считается напрямую из сохранённых дневных рядов
results/final/strategies/msp/perstrat_daily_returns/MSP_{A..D}_inst.csv
(колонка excess_ret, sqrt(252)). Бутстрап и CE - существующий скрипт
moex_volatility/scripts/fix_all_bootstrap_ce.py (только чтение, ~6 с).
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
MSP = ROOT / "results/final/strategies/msp/perstrat_daily_returns"


def main():
    for ap in "ABCD":
        d = pd.read_csv(MSP / f"MSP_{ap}_inst.csv", index_col=0)
        x = d.excess_ret.values
        sh = x.mean() / x.std(ddof=1) * np.sqrt(252)
        print(f"MSP-{ap}: XS Sharpe={sh:.4f} дней={len(d)}")
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/fix_all_bootstrap_ce.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    ).stdout
    keep = ("C>A", "D>A", "B>A", "D>C", "Common dates", "MSP-A > IMOEX")
    for line in out.splitlines():
        if any(k in line for k in keep):
            print(line.strip())


if __name__ == "__main__":
    main()
