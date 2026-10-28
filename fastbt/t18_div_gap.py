"""T18 — закрытие дивидендного гэпа (контракт #376 §2.4; толкование [Ж-065 п. 4]). Лонг.

Отсечка из таблицы дивидендов (`div`: ticker, last_cum_date, announce_date, dps_rub). g — первый торговый день после
last_cum_date (день гэпа). PIT: announce_date ≤ date_g, иначе (и без даты) события нет. Решение на close g, вход
close g+1 = «d0 + 2 торговых» (`postroit_sig`, lag = 1). gap_σ = (D / c_{d0}) / σ̂_{d0}, D — полный дивиденд, d0 = g−1.
Условие gap_σ ≥ g0. Сила = gap_σ (в σ).

inp = {"dates", "div": DataFrame, "tickers": {t: {"close", "sigma"}}}; osi = {"g0"}. Даты ≥ 2022 → PermissionError.
Таблицу читает вызывающий (`docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv`, строки < 2022); модуль файлов не
открывает. (`fastbt/dividendy.py` на базе ещё нет — #370; здесь та же логика g: первый бар > last_cum_date.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUT = np.datetime64("2022-01-01")
ZAD = 2                       # вход через ZAD торговых дней после d0; 1 — мутация (вход на g)


def sobytiya(inp: dict, osi: dict) -> list:
    dates = np.asarray(inp["dates"], "datetime64[D]")
    if len(dates) and dates.max() >= CUT:
        raise PermissionError("T18: даты ≥ 2022")
    g0 = float(osi["g0"])
    div = inp["div"]
    out = []
    for t in sorted(inp["tickers"]):
        x = inp["tickers"][t]
        c, sg = np.asarray(x["close"], float), np.asarray(x["sigma"], float)
        for row in div[div["ticker"] == t].itertuples():
            cum, ann = pd.Timestamp(row.last_cum_date), pd.Timestamp(row.announce_date)
            if pd.isna(ann) or cum.to_datetime64() >= CUT:
                continue
            g = int(np.searchsorted(dates, cum.to_datetime64(), side="right"))   # первый торговый день > last_cum_date
            d0 = g - 1
            if g < 1 or g >= len(dates) or ann.to_datetime64() > dates[g]:
                continue
            if not (c[d0] > 0 and sg[d0] > 0 and row.dps_rub > 0):
                continue
            gap = float(row.dps_rub) / c[d0] / sg[d0]
            if gap >= g0:
                out.append((t, d0 + ZAD - 1, 1, gap))
    return sorted(out, key=lambda e: (e[0], e[1]))
