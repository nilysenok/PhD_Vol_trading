#!/usr/bin/env python3
"""Восстановление cbr_key_rate.parquet (исходного файла нет на Маке).

2016-2021: key_rate_cbr_2016_2025.csv (disser). 2022-2025: ставка, обратно извлечённая из сохранённого
ряда MSP_A_inst.csv: rf = (total_ret - excess_ret) / cal_days * 365 (то, что вычитал исходный прогон).
Только чтение основной копии; выход data/strat_zamer/inputs/cbr_key_rate.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ms, w = Path(sys.argv[1]), Path(sys.argv[2])
d = pd.read_csv(ms / "results/final/strategies/msp/perstrat_daily_returns/MSP_A_inst.csv",
                index_col=0, parse_dates=True)
cal = np.r_[1, np.diff(d.index.values).astype("timedelta64[D]").astype(int)]
rf = ((d.total_ret - d.excess_ret) / cal * 365).round(4)
k = pd.read_csv("/Users/nilysenok/Desktop/Старое/disser/data/key_rate_cbr_2016_2025.csv",
                parse_dates=["date"]).set_index("date").rate.sort_index()
k = k[k.index < "2022-01-01"]
out = pd.concat([k, rf]).rename("key_rate").rename_axis("date").reset_index()
out = out.drop_duplicates("date", keep="last")
out.to_parquet(w / "data/strat_zamer/inputs/cbr_key_rate.parquet", index=False)
print(out.shape)
print(out.tail(2))
