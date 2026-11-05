"""Сверка собственной OLS HAR-J с arch.univariate.ARX (О9). arch читает те же пары и тот же x в exog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.univariate import ARX

from models_v2 import har


def arch_fit_forecast(Xt, yt, xnew):
    """(params[b0, b1..bk], ln f) из ARX с константой и exog; прогноз — штатный res.forecast."""
    res = ARX(yt, x=Xt, constant=True, rescale=False).fit(disp="off")
    k = Xt.shape[1]
    fc = res.forecast(horizon=1, x={"x%d" % j: np.array([[xnew[j]]]) for j in range(k)},
                      start=len(yt) - 1, reindex=False)
    return res.params.to_numpy()[:k + 1], float(fc.mean.to_numpy()[0, 0])


def check_dates(p: har.Panel, grid: np.ndarray, n: int = 10, cols=har.COLS_HARJ) -> pd.DataFrame:
    idx = grid[np.linspace(0, len(grid) - 1, n).round().astype(int)]
    rows = []
    for i in idx:
        Xt, yt = har.pairs(p, int(i), cols)
        beta, _, _ = har.ols(Xt, yt)
        x = p.X[i, list(cols)]
        ab, af = arch_fit_forecast(Xt, yt, x)
        lnf = float(beta[0] + beta[1:] @ x)
        rows.append({"d": p.dates[i], "n": len(yt), "d_lnf": abs(lnf - af),
                     "d_coef_abs": float(np.abs(beta - ab).max()),
                     "d_coef_rel": float((np.abs(beta - ab) / np.abs(beta)).max())})
    return pd.DataFrame(rows)
