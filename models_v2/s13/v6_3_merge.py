"""V6-3, G30: правило слияния R с A (план V6, п. 5, «Слияние с A»). На закрытых парах 2015-07-07 … 2016-12-29 (даты d; 377 дат — то же, что прогрев H) считается |корреляция Пирсона| каждого ρ с каждым из 12 признаков Σ.
ρ с максимальной |корреляцией| > 0,8 исключается из R до счёта E1 как дубль. Решение автоматическое, без ручного выбора; порог и окно — из плана."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_3_r as RF

WIN = ("2015-07-07", "2016-12-29")
THRESHOLD = 0.8


def merge_table(R: pd.DataFrame, sigma: pd.DataFrame, win: tuple = WIN, thr: float = THRESHOLD) -> tuple:
    """-> (таблица rho × признак Σ: corr, abs_corr, rho_max_abs, excluded; список оставшихся ρ). Окно — по датам d индекса."""
    ix = (R.index >= pd.Timestamp(win[0])) & (R.index <= pd.Timestamp(win[1]))
    rows = []
    for rho in RF.RHO_NAMES:
        for s in sigma.columns:
            c = float(R.loc[ix, rho].corr(sigma.loc[ix, s].reindex(R.index[ix])))
            rows.append(dict(rho=rho, sigma=s, n=int(ix.sum()), corr=c, abs_corr=abs(c)))
    t = pd.DataFrame(rows)
    mx = t.groupby("rho")["abs_corr"].max()
    t["rho_max_abs"] = t["rho"].map(mx)
    t["excluded"] = t["rho_max_abs"] > thr
    kept = [r for r in RF.RHO_NAMES if not bool(mx[r] > thr)]
    return t, kept
