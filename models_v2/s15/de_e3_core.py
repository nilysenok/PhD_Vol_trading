"""E3 (X3): устойчивая + адаптивная поправка. F = F_A0g·exp(a_t·g_a), где F_A0g = H·exp(g_s) — СОХРАНЁННАЯ A0g (g_s = ln(F_A0g/H); не переобучается: читается файл, sha256 — в CHECKPOINT).
Запись через F_A0g·exp(a·g_a) даёт при g_a = 0 ровно F = A0g побитно (exp(0) = 1.0). Чистые функции от массивов, файлов нет.
g_a = θ₀ + Z_s·θ — QLIKE-гребень с офсетом ln F_A0g на последних WIN = 250 закрытых парах (t⁺ ≤ d0): min Σ[u + R·e^(−u)] + λ·‖θ_Z‖², u = θ₀ + Z_s·θ, R = RV/F_A0g; θ₀ не штрафуется; выпуклая задача — seeds не нужны.
λ ∈ {1; 10} — внутренняя валидация: обучение на WIN закрытых парах, предшествующих последним HOLD = 126 закрытым парам, QLIKE на этих 126. Помесячная переоценка (первый торговый день месяца с 2017-01-01); мало строк (< MIN_TRAIN = 186) — g_a = 0.
a_t = σ((S_t − h)/s) — сигмоида (наклон 1, в единицах s) от одностороннего CUSUM S_t = max(0, S_{t−1} + δ_j), δ_j = QLIKE(F_A0g) − QLIKE(F_A0g·e^{g_a}) закрытых пар j (t⁺_j ≤ d; g_a — прогноз модели, действовавшей в день j),
по последним WIN закрытым парам с g_a; s — СКО δ на этих парах (ddof = 0), h = 5·s. Меньше MIN_DELTA = 20 пар или s ≈ 0 — a_t = 0."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HOLD, WIN, LAMS, MIN_TRAIN, FIRST_REFIT, H_MULT, SLOPE, MIN_DELTA, MAXITER = 126, 250, (1.0, 10.0), 186, "2017-01-01", 5.0, 1.0, 20, 300


def closed_rows(tp, d0) -> np.ndarray:
    """Закрытые пары на d0: t⁺ ≤ d0. МУТАНТ «закрыто + 1» подменяет эту функцию в тесте."""
    return np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0))


def qlike(rv, F):
    r = np.asarray(rv, float) / np.asarray(F, float)
    return r - np.log(r) - 1.0


def loss_grad(th, Xs, R, lam) -> tuple:
    th = np.asarray(th, float)
    Xc = np.hstack([np.ones((len(Xs), 1)), Xs])
    u = Xc @ th
    e = R * np.exp(-u)
    pen = th.copy()
    pen[0] = 0.0
    return float(np.sum(u + e) + lam * float(pen @ pen)), Xc.T @ (1.0 - e) + 2.0 * lam * pen


def ridge_fit(Xs, R, lam: float) -> np.ndarray:
    r = minimize(loss_grad, np.zeros(Xs.shape[1] + 1), args=(Xs, R, lam), jac=True, method="L-BFGS-B", options=dict(maxiter=MAXITER))
    return r.x


def apply(th, Xs) -> np.ndarray:
    return th[0] + Xs @ th[1:]


def norm_stats(X: np.ndarray) -> tuple:
    sd = X.std(axis=0)
    return X.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)


def refit(F0, rv, tp, Z, d0):
    """-> dict(th, mu, sd, lam, val, n_rows, rows) или None (мало строк). rows — индексы окна последнего обучения."""
    ok = np.isfinite(Z).all(axis=1) & np.isfinite(rv) & np.isfinite(F0) & (F0 > 0)
    cl_all = closed_rows(tp, d0)
    cl = cl_all[ok[cl_all]]
    if len(cl) < MIN_TRAIN:
        return None
    va, tr = cl[-HOLD:], cl[:-HOLD][-WIN:]
    R = rv / F0
    mu, sd = norm_stats(Z[tr])
    val = {}
    for lam in LAMS:
        th = ridge_fit((Z[tr] - mu) / sd, R[tr], lam)
        val[lam] = float(np.mean(qlike(rv[va], F0[va] * np.exp(apply(th, (Z[va] - mu) / sd)))))
    lam = min(LAMS, key=lambda k: val[k])
    rows = cl[-WIN:]
    mu, sd = norm_stats(Z[rows])
    return dict(th=ridge_fit((Z[rows] - mu) / sd, R[rows], lam), mu=mu, sd=sd, lam=lam, val=val, n_rows=len(rows), rows=rows)


def refit_positions(index: pd.DatetimeIndex, first: str = FIRST_REFIT) -> list:
    per = index.to_period("M")
    return [int(np.flatnonzero(per == m)[0]) for m in per.unique() if m.start_time >= pd.Timestamp(first)]


def walk_forward_ga(F0, rv, tp, dates, Z) -> dict:
    """-> dict(ga, fitted, lam, n_rows, fits). До первой модели g_a = 0. Прогноз дня d — моделью последней переоценки ≤ d."""
    F0, rv, Z, dates = np.asarray(F0, float), np.asarray(rv, float), np.asarray(Z, float), pd.DatetimeIndex(dates)
    n, pos, fits = len(F0), refit_positions(dates), []
    out = dict(ga=np.zeros(n), fitted=np.zeros(n, bool), lam=np.full(n, np.nan), n_rows=np.zeros(n, int))
    for a, b in zip(pos, pos[1:] + [n]):
        r = refit(F0, rv, tp, Z, dates[a])
        fits.append(dict(d0=dates[a], fitted=r is not None, lam=None if r is None else r["lam"], n_rows=0 if r is None else r["n_rows"]))
        if r is None:
            continue
        ok = np.isfinite(Z[a:b]).all(axis=1)
        out["ga"][a:b] = np.where(ok, apply(r["th"], (np.where(ok[:, None], Z[a:b], 0.0) - r["mu"]) / r["sd"]), 0.0)
        out["fitted"][a:b] = ok
        out["lam"][a:b], out["n_rows"][a:b] = r["lam"], r["n_rows"]
    return dict(fits=fits, **out)


def cusum_state(delta: np.ndarray) -> tuple:
    """-> (S_t на конце ряда, s = СКО δ ddof = 0). S_t = max(0, S + δ), старт 0."""
    s_ = 0.0
    for x in delta:
        s_ = max(0.0, s_ + x)
    return s_, float(np.std(delta))


def adaptive_weights(rv, F0, ga, fitted, tp, dates) -> np.ndarray:
    """a_t в [0; 1] по закрытым парам на дне d = dates[i]; меньше MIN_DELTA пар или s ≈ 0 — 0."""
    rv, F0, ga, dates = np.asarray(rv, float), np.asarray(F0, float), np.asarray(ga, float), pd.DatetimeIndex(dates)
    delta = np.where(fitted, qlike(rv, F0) - qlike(rv, F0 * np.exp(ga)), np.nan)
    a = np.zeros(len(F0))
    for i, d in enumerate(dates):
        cl = closed_rows(tp, d)
        j = cl[fitted[cl] & np.isfinite(delta[cl])][-WIN:]
        if len(j) < MIN_DELTA:
            continue
        S, s = cusum_state(delta[j])
        if s > 1e-12:
            a[i] = float(expit(SLOPE * (S - H_MULT * s) / s))
    return a


def forecast(F0, ga, a) -> np.ndarray:
    """F = F_A0g·exp(a·g_a); g_a = 0 ⇒ F = F_A0g побитно."""
    return np.asarray(F0, float) * np.exp(np.asarray(a, float) * np.asarray(ga, float))
