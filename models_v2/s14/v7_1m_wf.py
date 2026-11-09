"""V7-1a-M, walk-forward (план п. 4, Ж-233 п. 2). В первый торговый день каждого месяца (`refit_positions`, V6) голова оценивается ТОЛЬКО на закрытых парах t⁺ ≤ d0; последние 126 закрытых — внутренняя валидация (`val_rows`),
остальные — обучение; Z и z стандартизуются по среднему и СКО закрытых пар (`norm_stats`); прогноз дня d — моделью последней переоценки ≤ d. 2017 — только обучение: первая оцениваемая дата 2018-01-03 (до неё F = NaN).
Мало закрытых строк (< MIN_TRAIN) — модели нет: G = 0, F = H, fitted = False. Stage B (совместное): до первой переоценки `first_fit` (2019-01-01) берётся прогноз Stage A той же конфигурации (`fallback`), с пометкой stage = 'A'.
Строки с нечисловым Z/z в обучение не входят. Модуль не читает данные: таблица и признаки приходят аргументами."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1m_fit as FT          # torch раньше lightgbm (v6_1_wf -> v6_1_fit): на macOS общий OpenMP; все вычисления torch — в одном потоке (`FM.deterministic`)
from models_v2.s13 import v6_1_wf as W6

FIRST_EVAL, FIRST_REFIT, FIRST_FIT_B = "2018-01-03", "2017-01-01", "2019-01-01"
HOLD, MIN_TRAIN = 126, 126 + 60


def closed_rows(t_plus, d0) -> np.ndarray:
    """Закрытые пары на d0: t⁺ ≤ d0 (V6). МУТАНТ «закрыто + 1» подменяет эту функцию в тесте."""
    return W6.closed_rows(t_plus, d0)


def val_rows(t_plus, d0, usable: np.ndarray) -> np.ndarray:
    """Внутренняя валидация: последние HOLD пригодных закрытых пар. МУТАНТ «с парами t⁺ > d» подменяет эту функцию."""
    r = closed_rows(t_plus, d0)
    return r[usable[r]][-HOLD:]


def norm_stats(X: np.ndarray) -> tuple:
    """Среднее и СКО (ddof = 0) по строкам закрытых пар. МУТАНТ «по всей выборке» подменяет эту функцию."""
    sd = X.std(axis=0)
    return X.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)


def _usable(Zg, z) -> np.ndarray:
    ok = np.isfinite(Zg).all(axis=1)
    return ok & np.isfinite(z.reshape(len(z), -1)).all(axis=1)


def _prep_fn(Zg, z, ell, H, rv, rows, joint):
    """Стандартизация по строкам `rows` (закрытые пары). В совместном режиме слот z (последовательности) не стандартизуется."""
    A = Zg if joint else np.hstack([Zg, z])
    mu, sd = norm_stats(A[rows])
    k = Zg.shape[1]

    def prep(idx):
        zz = z[idx] if joint else (z[idx] - mu[k:]) / sd[k:]
        return FT.batch((Zg[idx] - mu[:k]) / sd[:k], zz, ell[idx], H[idx], rv[idx])
    return prep


def _refit(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid):
    H, rv, tp, ell = tab_cols
    usable = _usable(Zg, z)
    cl = closed_rows(tp, d0)
    cl = cl[usable[cl]]
    va = val_rows(tp, d0, usable)
    tr = np.setdiff1d(cl, va)
    rows = np.union1d(tr, va)
    if len(rows) < MIN_TRAIN:
        return None, len(rows)
    prep = _prep_fn(Zg, z, ell, H, rv, rows, spec.get("encoder") is not None)
    r = FT.fit_config(spec, prep(tr), prep(va), seeds, grid)
    lnf = FT.ensemble_lnF(r["models"], prep(np.arange(lo, b))).numpy()               # внутри — `FM.deterministic()` (один поток)
    return dict(G=lnf - np.log(H[lo:b]), lam=r["lam"], val=r["val"], epochs=r["epochs"]), len(rows)


def walk_forward(tab: pd.DataFrame, Zg: np.ndarray, z: np.ndarray, spec: dict, seeds=FT.SEEDS, grid=FT.GRID, first_eval: str = FIRST_EVAL,
                 first_fit: str | None = None, fallback: pd.Series | None = None) -> dict:
    """tab: index d, колонки H, K2, K3, rv_next, t_plus. Zg (n, 9) — сырые признаки режима; z (n, 4) — сырой вход слота z (в совместном — последовательности).
    -> {F, G, fitted, stage: Series; lam (n, 2); fits: список переоценок}. Позиции переоценок — первый торговый день месяца."""
    H, rv, tp = tab["H"].to_numpy(float), tab["rv_next"].to_numpy(float), tab["t_plus"].to_numpy("datetime64[ns]")
    ell = np.log(np.sqrt(tab["K2"].to_numpy(float) * tab["K3"].to_numpy(float)) / H)
    n, pos = len(tab), W6.refit_positions(tab.index, FIRST_REFIT)
    G, fitted, lam = np.full(n, np.nan), np.zeros(n, bool), np.full((n, 2), np.nan)
    stage = np.array(["B" if spec.get("encoder") is not None else "A"] * n)
    i0, fits = int(np.searchsorted(tab.index.to_numpy("datetime64[ns]"), np.datetime64(first_eval))), []
    for a, b in zip(pos, pos[1:] + [n]):
        if b <= i0:
            continue
        lo, d0 = max(a, i0), tab.index[a]
        res, nrow = (None, 0) if (first_fit and d0 < pd.Timestamp(first_fit)) else _refit((H, rv, tp, ell), Zg, z, spec, d0, lo, b, seeds, grid)
        if res is None:
            G[lo:b] = 0.0
            fits.append(dict(d0=d0, n_train=nrow, fitted=False))
            continue
        G[lo:b], fitted[lo:b], lam[lo:b] = res["G"], True, res["lam"]
        fits.append(dict(d0=d0, n_train=nrow, fitted=True, lam=res["lam"], val=res["val"], epochs=res["epochs"]))
    F = H * np.exp(G)
    if fallback is not None:                                                   # Stage B до первой переоценки — прогноз Stage A
        pre = ~fitted & (tab.index >= pd.Timestamp(first_eval))
        F[pre], stage[pre] = fallback.reindex(tab.index).to_numpy(float)[pre], "A"
    return dict(F=pd.Series(F, index=tab.index, name="F"), G=pd.Series(G, index=tab.index), fitted=pd.Series(fitted, index=tab.index), stage=pd.Series(stage, index=tab.index), lam=lam, fits=fits)
