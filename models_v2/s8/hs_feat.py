"""Ж-212 (HS5): PIT-коэффициенты HAR-J и блоки признаков B на строках F (все строки, включая до сетки). Строка t видит только данные <= t;
прошлые ошибки и расхождения — только закрытые прогнозы (target_date_j = дата строки j+1 <= дата t), `hs_pit`. `extra` — подложенная ошибка «закрыто + extra»."""
from __future__ import annotations

import numpy as np

from models_v2 import har, metrics
from models_v2.s2 import features as ft
from models_v2.s8 import hs_pit

FAR = np.datetime64("2100-01-01", "ns")
BLOCKS = ("Bf", "Bc", "Br", "Bg", "Be")


def har_pit(F, min_pairs: int = ft.MIN_PAIRS_HAR) -> tuple:
    """-> (lb, beta): lb_t = ln σ̂²_b HAR-J по парам t'+1 <= t (как `resid.harj_ab`), beta (n×5: const, d, w, m, J) — коэффициенты той же подгонки; NaN, где < min_pairs."""
    p = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    n = len(p.rv)
    lb, beta = np.full(n, np.nan), np.full((n, 5), np.nan)
    for i in range(n):
        if not np.isfinite(p.X[i]).all():
            continue
        Xt, yt = har.pairs(p, i, har.COLS_HARJ)
        if len(yt) < min_pairs:
            continue
        b, s2, _ = har.ols(Xt, yt)
        beta[i] = b
        lb[i] = float(b[0] + (b[1:] * p.X[i]).sum()) + s2 / 2
    return lb, beta


def dates_td(F) -> tuple:
    d = np.asarray(F.dates, "datetime64[ns]")
    return d, np.append(d[1:], FAR)


def har_errors(F, lb) -> np.ndarray:
    """e_j = ln RV_{j+1} − lb_j (известна с даты строки j+1)."""
    e = np.full(len(lb), np.nan)
    e[:-1] = F.lnrv[1:] - lb[:-1]
    return e


def block_f(lb) -> tuple:
    return lb[:, None], ["Bf_lnHAR"]


def block_c(F) -> tuple:
    return F.X[:, :4].copy(), ["Bc_lnRVd", "Bc_lnRVw", "Bc_lnRVm", "Bc_ln1pJ"]


def block_r(F, beta) -> tuple:
    """s_k = β̂_k φ_k / Σ|β̂_j φ_j| (β̂ PIT), ln RV_d − ln RV_m, ln RV_w − ln RV_m, ln(RS−/RS+)."""
    phi = F.X[:, :4]
    contrib = beta[:, 1:] * phi
    s = contrib / np.abs(contrib).sum(axis=1, keepdims=True)
    asym = np.log((F.X[:, 8] + 1e-12) / (F.X[:, 7] + 1e-12))
    cols = np.column_stack([s, phi[:, 0] - phi[:, 2], phi[:, 1] - phi[:, 2], asym])
    return cols, ["Br_s_d", "Br_s_w", "Br_s_m", "Br_s_J", "Br_d_m", "Br_w_m", "Br_asym"]


def block_g(F, lb, ml_rows, extra: int = 0) -> tuple:
    """Прошлое расхождение HAR−ML по закрытым: средние δ и dq за 22 и 63. ml_rows — σ̂² ML на строках F (NaN вне сетки)."""
    d, td = dates_td(F)
    har_s2 = np.exp(lb)
    delta = np.log(har_s2 / ml_rows)
    rv1 = np.append(F.rv[1:], np.nan)
    dq = metrics.qlike(rv1, har_s2) - metrics.qlike(rv1, ml_rows)
    cols = [hs_pit.past_mean(x, d, td, w, extra=extra) for x in (delta, dq) for w in (22, 63)]
    return np.column_stack(cols), ["Bg_delta22", "Bg_delta63", "Bg_dq22", "Bg_dq63"]


def block_e(F, lb, extra: int = 0) -> tuple:
    """Прошлые ошибки HAR: последняя, среднее 5 и 22 закрытых."""
    d, td = dates_td(F)
    e = har_errors(F, lb)
    cols = [hs_pit.past_last(e, d, td, extra), hs_pit.past_mean(e, d, td, 5, extra=extra), hs_pit.past_mean(e, d, td, 22, extra=extra)]
    return np.column_stack(cols), ["Be_last", "Be_m5", "Be_m22"]


def blocks(F, lb, beta, ml_rows, extra: int = 0) -> dict:
    return {"Bf": block_f(lb), "Bc": block_c(F), "Br": block_r(F, beta), "Bg": block_g(F, lb, ml_rows, extra), "Be": block_e(F, lb, extra)}


def state_c(F, lb, extra: int = 0) -> tuple:
    """Cc: «состояния HAR» (7): ln(RV_w/RV_m), ln(RV_d/RV_w), ln(1+J), асимметрия RS, e_HAR последняя / среднее 5 / среднее 22 закрытых."""
    be, nb = block_e(F, lb, extra)
    x = F.X
    cols = np.column_stack([x[:, 1] - x[:, 2], x[:, 0] - x[:, 1], x[:, 3], np.log((x[:, 8] + 1e-12) / (x[:, 7] + 1e-12)), be])
    return cols, ["Cc_w_m", "Cc_d_w", "Cc_J", "Cc_asym"] + nb


def ml_on_rows(n: int, grid, s2) -> np.ndarray:
    out = np.full(n, np.nan)
    out[np.asarray(grid, int)] = np.asarray(s2, float)
    return out
