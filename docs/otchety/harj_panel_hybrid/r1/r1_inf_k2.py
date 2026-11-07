"""Р-1, находка 2: ячейки ±inf в сыром дизайне K2 (Б-2 + F4) по 28 бумагам, источник и причина (код/данные не меняются; только чтение).

Запуск (из корня): python3 -W ignore docs/otchety/harj_panel_hybrid/r1/r1_inf_k2.py [тикер ...]   (без аргументов — все P.TICKERS)
Печатает: (1) нефинитные ячейки (±inf) в F.X (признаки мира), dz.X (дизайн K2 = worldx.spec_of(CTRL).X); имя столбца, дата;
(2) для каждой — дневные величины в ту дату (n_bars, rv, bv, j) и число ненулевых 10-минутных |r| основной сессии.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from models_v2 import daily  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s10 import hp_k23 as K, hp_k23_world as KW, hp_paths as P  # noqa: E402
from models_v2.s6.xgb import cfgx, worldx  # noqa: E402


def scan(t: str, imoex, pool) -> None:
    bars = K.read_bars(t)
    w = KW.build_world(bars, imoex, pool)
    F = w["F"]
    X = np.asarray(worldx.spec_of(w, cfgx.CTRL).X, float)
    n0 = w["dz0"].X.shape[1]
    nm = list(ft.NAMES[:n0]) if n0 <= len(ft.NAMES) else [str(j) for j in range(n0)]
    print("%s: строк %d, столбцов K2 %d (Б-2: %d), ±inf в F.X: %d, ±inf в X K2: %d" % (
        t, len(F.dates), X.shape[1], n0, int(np.isinf(F.X).sum()), int(np.isinf(X).sum())))
    for i, j in zip(*np.where(np.isinf(X))):
        col = "j=%d (%s)" % (j, nm[j] if j < n0 else "F4-признак") if j < n0 else "j=%d (F4)" % j
        print("   ячейка: дата %s, столбец %s, значение %s" % (pd.Timestamp(F.dates[i]).date(), col, X[i, j]))
        if j < ft.N_BASE:
            print("   сырой F.X[:, %d] в этой строке: %s" % (j, F.X[i, j]))
        d, _ = daily.build_daily(bars)
        print("   дневная строка:", d.loc[pd.Timestamp(F.dates[i]), ["n_bars", "rv", "bv", "j"]].to_dict())
        main = daily.main_session(bars)
        day = pd.Timestamp(F.dates[i])
        b = main[main["begin"].dt.normalize() == day]
        r = daily.intraday_returns(b.sort_values("begin").reset_index(drop=True))["r"].to_numpy()
        print("   баров основной сессии %d, нулевых r %d, пар подряд с ненулевыми r (ненулевая BV): %d" % (
            len(b), int((r == 0).sum()), int(((r[1:] != 0) & (r[:-1] != 0)).sum())))
        print("   объём: мин %s, баров с объёмом 0: %d; close: %s" % (b["volume"].min(), int((b["volume"] == 0).sum()), sorted(set(np.round(b["close"], 4)))[:6]))


if __name__ == "__main__":
    tick = sys.argv[1:] or P.TICKERS
    imoex = K.read_bars("IMOEX")
    pool = KW.pool_papers(K.read_bars)
    for t in tick:
        scan(t, imoex, pool)
