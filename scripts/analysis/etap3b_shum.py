#!/usr/bin/env python3
"""etap3b: шум одной строкой. RV(10)/RV(60) (отношение средних дневных RV, затем среднее по 17 тикерам) и
доля скачков J/RV на 10 и 30 минутах; BV = (pi/2)*sum |r_i||r_{i-1}| (Barndorff-Nielsen-Shephard) внутри дня,
J = max(RV - BV, 0); доля = среднее по дням ежедневных J/RV, затем среднее по тикерам. Остановка 28.02-24.03.2022 исключена.
Запуск: python3 scripts/analysis/etap3b_shum.py"""
import numpy as np, pandas as pd
from etap3b_signature import CANDLES, TICKERS, HALT


def one(tk):
    d = pd.read_parquet(CANDLES / f"{tk}.parquet", columns=["timestamp", "close"])
    d = d[d.close > 0].sort_values("timestamp"); d["day"] = d.timestamp.dt.normalize()
    d = d[(d.day < HALT[0]) | (d.day > HALT[1])]; d["lc"] = np.log(d.close); d["k"] = d.groupby("day").cumcount()
    out = {}
    for f in (10, 30, 60):
        s = d[d.k % (f // 10) == 0].copy(); s["r"] = s.groupby("day").lc.diff(); s = s.dropna(subset=["r"])
        s["a"] = s.r.abs(); s["bv"] = np.pi / 2 * s.a * s.groupby("day").a.shift(1)
        g = s.groupby("day"); rv = (s.r ** 2).groupby(s.day).sum(); bv = g.bv.sum()
        out[f] = (rv, (rv - bv).clip(lower=0) / rv)
    return out[10][0].mean() / out[60][0].mean(), out[10][1].mean(), out[30][1].mean()


def main():
    r = np.array([one(t) for t in TICKERS]).mean(axis=0)
    ru = lambda v: f"{v:.4f}".replace(".", ",")
    print(f"шум: RV(10 мин)/RV(60 мин) = {ru(r[0])}; доля скачков J/RV: 10 мин {ru(100*r[1])[:-2]}%, 30 мин {ru(100*r[2])[:-2]}% (17 тикеров)")


if __name__ == "__main__":
    main()
