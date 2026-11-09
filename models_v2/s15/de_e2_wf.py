"""E2 (X2): walk-forward (как V7-1a-M). В первый торговый день каждого месяца (с 2017-01-01) модель оценивается ТОЛЬКО на закрытых парах t⁺ ≤ d0; последние HOLD = 126 пригодных закрытых пар — внутренняя
валидация (выбор β ∈ {0,1; 1} по QLIKE ансамбля 5 seeds), затем 5 seeds переобучаются на всех закрытых парах с выбранным β; признаки стандартизуются по среднему/СКО (ddof = 0) этих же закрытых пар.
Метка стресса — по порогу закрытых пар d0. Прогноз дня d — моделью последней переоценки ≤ d. Мало строк (< MIN_TRAIN) — F = H, fitted = False. Данные не читает: массивы приходят аргументами."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de_e2_feat as FE, de_e2_model as M

HOLD, MIN_TRAIN, FIRST_REFIT = 126, 126 + 60, "2017-01-01"


def closed_rows(tp, d0) -> np.ndarray:
    """Закрытые пары на d0: t⁺ ≤ d0. МУТАНТ «закрыто + 1» подменяет эту функцию в тесте."""
    return np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0))


def val_rows(tp, d0, usable: np.ndarray) -> np.ndarray:
    r = closed_rows(tp, d0)
    return r[usable[r]][-HOLD:]


def norm_stats(X: np.ndarray) -> tuple:
    sd = X.std(axis=0)
    return X.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)


def refit_positions(index: pd.DatetimeIndex, first: str = FIRST_REFIT) -> list:
    per = index.to_period("M")
    return [int(np.flatnonzero(per == m)[0]) for m in per.unique() if m.start_time >= pd.Timestamp(first)]


def refit(H, rv, tp, Xraw, d0, seeds, betas, maxiter):
    """-> dict(models, beta, thr, mu, sd, val) или None (мало строк)."""
    usable = np.isfinite(Xraw).all(axis=1) & np.isfinite(rv) & np.isfinite(H)
    cl_all = closed_rows(tp, d0)
    cl = cl_all[usable[cl_all]]
    va = val_rows(tp, d0, usable)
    tr = np.setdiff1d(cl, va)
    rows = np.union1d(tr, va)
    if len(rows) < MIN_TRAIN:
        return None
    thr = FE.stress_threshold(rv, cl_all)
    y, R = FE.stress_label(rv, thr), rv / H
    mu, sd = norm_stats(Xraw[rows])
    X = (Xraw - mu) / sd
    val = {b: M.qlike_val(R[va], M.predict([M.fit(X[tr], R[tr], y[tr], b, s, maxiter) for s in seeds], X[va])["lnm"]) for b in betas}
    beta = min(betas, key=lambda b: val[b])
    return dict(models=[M.fit(X[rows], R[rows], y[rows], beta, s, maxiter) for s in seeds], beta=beta, thr=thr, mu=mu, sd=sd, val=val, n_rows=len(rows))


def walk_forward(H, rv, tp, dates, Xraw, first_eval: str, seeds=M.SEEDS, betas=M.BETAS, maxiter: int = M.MAXITER) -> dict:
    """-> dict(F, p, cn, cs, beta, thr, fitted, fits). Массивы по всем строкам таблицы; до first_eval и без модели: F = H, p = NaN."""
    H, rv, Xraw = np.asarray(H, float), np.asarray(rv, float), np.asarray(Xraw, float)
    n, dates = len(H), pd.DatetimeIndex(dates)
    pos = refit_positions(dates)
    out = dict(lnm=np.zeros(n), p=np.full(n, np.nan), cn=np.full(n, np.nan), cs=np.full(n, np.nan), beta=np.full(n, np.nan), thr=np.full(n, np.nan), fitted=np.zeros(n, bool))
    i0, fits = int(np.searchsorted(dates.to_numpy("datetime64[ns]"), np.datetime64(first_eval))), []
    for a, b in zip(pos, pos[1:] + [n]):
        if b <= i0:
            continue
        lo = max(a, i0)
        r = refit(H, rv, tp, Xraw, dates[a], seeds, betas, maxiter)
        fits.append(dict(d0=dates[a], fitted=r is not None, beta=None if r is None else r["beta"], val=None if r is None else r["val"]))
        if r is None:
            continue
        pr = M.predict(r["models"], (Xraw[lo:b] - r["mu"]) / r["sd"])
        out["lnm"][lo:b] = pr["lnm"]
        for k in ("p", "cn", "cs"):
            out[k][lo:b] = pr[k]
        out["beta"][lo:b], out["thr"][lo:b], out["fitted"][lo:b] = r["beta"], r["thr"], True
    return dict(F=H * np.exp(out.pop("lnm")), fits=fits, **out)
