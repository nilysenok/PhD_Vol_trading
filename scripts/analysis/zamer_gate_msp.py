#!/usr/bin/env python3
"""Ворота: Шарп МСП A/B/C/D из песочницы против сохранённых рядов основной копии (чтение).
Запуск: python3 zamer_gate_msp.py <sbx/moex_volatility>"""
import sys
from pathlib import Path

import pandas as pd

sbx = Path(sys.argv[1])
old_dir = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility/results/final/strategies/msp/perstrat_daily_returns")
new_dir = sbx / "results/final/strategies/msp/perstrat_daily_returns"
sh = lambda d: d.excess_ret.mean() / d.excess_ret.std(ddof=1) * 252 ** 0.5
for a in "ABCD":
    new = pd.read_csv(new_dir / f"MSP_{a}_inst.csv", index_col=0)
    old = pd.read_csv(old_dir / f"MSP_{a}_inst.csv", index_col=0)
    md = (new.excess_ret - old.excess_ret).abs().max() if len(new) == len(old) and (new.index == old.index).all() else None
    print(f"MSP-{a}: новый {sh(new):.4f} сохранённый {sh(old):.4f} дней {len(new)}/{len(old)} max|d excess|={md}")
