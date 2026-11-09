"""V6-2, G24/G25: помесячная переоценка с индексом u и взаимодействиями u·(s1, s2, s3). Та же схема, что у движка листа 1a (`v6_1_wf.walk_forward`: первый торговый день месяца, закрытые пары t⁺ ≤ d0, расширяющееся окно),
но u зависит от даты переоценки: нормировка его компонент и порог верхнего квинтиля считаются по закрытым парам d0 (PIT), затем матрица X = S + Q (+ u·s1…s3) строится на всех строках. Нормировка самих признаков — в обучателе (по тем же парам).
Движок листов 1a/1b не менялся: используются его `closed_rows`, `refit_positions`, `residual`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_wf as WF, v6_2_q as QF

INTER = ["u_s1", "u_s2", "u_s3"]


def design(S: np.ndarray, Q: np.ndarray, tr: np.ndarray, interactions: bool) -> tuple:
    """-> (X, u, порог квинтиля) на дату переоценки по закрытым парам tr."""
    u = QF.u_values(Q, tr)
    X = [S, Q] + ([u[:, None] * S[:, :3]] if interactions else [])
    return np.column_stack(X), u, QF.u_threshold(u, tr)


def names(S: pd.DataFrame, interactions: bool) -> list:
    return list(S.columns) + QF.Q_NAMES + (INTER if interactions else [])


def walk_forward_u(tab: pd.DataFrame, S: pd.DataFrame, Qd: pd.DataFrame, fit, interactions: bool = False, first: str = WF.FIRST_REFIT) -> dict:
    """-> {g, u, thr, high (Series по дням таблицы; до первой переоценки g = 0, u/thr = NaN), fits}. fit=None — только группы дня (без обучения)."""
    Sv, Qv, e, tp = S.to_numpy(float), Qd.reindex(S.index).to_numpy(float), WF.residual(tab), tab["t_plus"].to_numpy()
    n = len(tab)
    g, u, thr = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    pos, fits = WF.refit_positions(tab.index, first), []
    for a, b in zip(pos, pos[1:] + [n]):
        tr = WF.closed_rows(tp, tab.index[a])
        X, uu, th = design(Sv, Qv, tr, interactions)
        u[a:b], thr[a:b] = uu[a:b], th
        if fit is not None:
            m = fit(X[tr], e[tr])
            g[a:b] = m.g(X[a:b])
            fits.append(dict(d0=tab.index[a], n_train=len(tr), model=m))
    ix = tab.index
    return dict(g=pd.Series(g, index=ix, name="g"), u=pd.Series(u, index=ix, name="u"), thr=pd.Series(thr, index=ix, name="thr"), high=pd.Series(u > thr, index=ix, name="high"), fits=fits)
