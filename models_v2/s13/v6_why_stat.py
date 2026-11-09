"""V6 лист Z3: общие статистики. Бутстрап — реализация V6 FIN (`stats_d.block_idx/boot_means`, блок 20, 2000, seed 0, константы `v6_1_eval`), QLIKE — `v6_0_q.ql`, лаг и веса Бартлетта — `s3.dm` (импорт).
HAC-ошибки регрессии: матричный вариант Ньюи–Уэста с теми же весами и лагом 5, что у `dm.nw_variance` (там — для среднего; при единственном регрессоре-константе совпадает с ним, тест)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as DM
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV


def dd(rv, f_hyb, f_base) -> np.ndarray:
    """Дневная разность QLIKE: l(гибрид) − l(база); минус = гибрид лучше."""
    return Q.ql(rv, f_hyb) - Q.ql(rv, f_base)


def year_masks(tp) -> dict:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    return {str(k): y == k for k in range(2017, 2027)}


def pct(x: float, level: float) -> float:
    """В процентах уровня QLIKE K1."""
    return 100.0 * float(x) / float(level)


def hac_ols(y, X, lag: int = DM.LAG) -> tuple:
    """OLS y = Xβ + e; -> (β, se) с HAC (Бартлетт, лаг как в DM)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    XtXi = np.linalg.inv(np.einsum("ni,nj->ij", X, X))              # einsum, не matmul: на macOS (Accelerate) matmul даёт ложные RuntimeWarning при конечном результате
    beta = np.einsum("ij,j->i", XtXi, np.einsum("ni,n->i", X, y))
    u = X * (y - np.einsum("ni,i->n", X, beta))[:, None]
    S = np.einsum("ni,nj->ij", u, u)
    for l in range(1, lag + 1):
        G = np.einsum("ni,nj->ij", u[l:], u[:-l])
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = np.einsum("ij,jk,kl->il", XtXi, S, XtXi)
    return beta, np.sqrt(np.diag(V))


def boot_pct(x, level: float) -> np.ndarray:
    """Бутстрап-средние (2000, блок 20, seed 0) ряда x в % уровня."""
    z = 100.0 * np.asarray(x, float) / level
    return SD.boot_means(z, SD.block_idx(len(z), B_=EV.B, block=EV.BLOCK, seed=EV.SEED))


def diff_ci(x_early, k_early: float, x_late, k_late: float) -> dict:
    """Δ_поздний − Δ_ранний (в % уровня K1 своего окна) и 90 % интервал (5-й, 95-й перцентили) разности бутстрап-средних независимых окон."""
    be, bl = boot_pct(x_early, k_early), boot_pct(x_late, k_late)
    d = bl - be
    e, l = 100.0 * float(np.mean(x_early)) / k_early, 100.0 * float(np.mean(x_late)) / k_late
    return dict(delta_early_pct=e, delta_late_pct=l, diff_pct=l - e, lo90=float(np.percentile(d, 5)), hi90=float(np.percentile(d, 95)))
