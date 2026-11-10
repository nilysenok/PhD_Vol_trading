"""ZD (Ж-246, план Z раздел 7): предсказуема ли полезность поправки Y заранее. ДИАГНОСТИКА: прогнозов RV, конфигураций, строк реестра нет; ничего не пишет.
D_t = l(Y, t) − l(X1, t), l = RV/F − ln(RV/F) − 1. Признаки закрытия дня d: z_d = (e_d, x_d, s_d, v_d), только бары дня d и ПРОШЛЫЕ дни (v_d — 20 прошлых RV).
Помесячно с 2018-07-02, пул — пары с t⁺ ≤ D (`x1_ya_walk.closed_pool`); стандартизация z̃ — по датам пула. Модели: (а) логистическая 1[D<0] (L2, C = 1), (б) гребень D (α = 1), (в) только e_d, (г) (а) на z_{d−252} (циклический сдвиг)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models_v2.s16 import x1_ya_walk as W

LAG, RIDGE_ALPHA, LOGIT_C, LOOKBACK = 252, 1.0, 1.0, 20
ZNAMES = ["e", "x", "s", "v"]


def qlike(rv, F) -> np.ndarray:
    u = np.asarray(rv, float) / np.asarray(F, float)
    return u - np.log(u) - 1.0


def dloss(rv, F, X1) -> np.ndarray:
    """D_t = l(Y) − l(X1): D < 0 — поправка помогла."""
    return qlike(rv, F) - qlike(rv, X1)


def zrow(p: int) -> int:
    """Строка склада бар-признаков дня d: СТРОГО день d (мутация: p + 1 — бары дня t⁺)."""
    return p


def day_xv(r2: np.ndarray, present: np.ndarray, past_rv: np.ndarray) -> tuple:
    """x_d = ln max r² − ln(RV_d/n_d); v_d = ln RV_d − ln mean(RV прошлых 20 дней). Вырожденные (нет баров, нулевая RV) → 0."""
    x = np.where(present, r2, 0.0)
    n, rv = int(present.sum()), float(x.sum())
    xd = float(np.log(x.max()) - np.log(rv / n)) if n and rv > 0 and x.max() > 0 else 0.0
    m = float(np.mean(past_rv)) if len(past_rv) else 0.0
    vd = float(np.log(rv) - np.log(m)) if rv > 0 and m > 0 else 0.0
    return xd, vd


def zfeatures(store: dict, d, e, s) -> np.ndarray:
    """(n, 4): e, x, s, v. store: days, r2, present (склад M10 V7); d — даты прогнозов; e, s — колонка s x1.parquet и s_d Y-A."""
    pos = pd.DatetimeIndex(store["days"]).get_indexer(pd.DatetimeIndex(d))
    if (pos < 0).any():
        raise SystemExit("день d вне склада M10")
    rv_all = np.where(store["present"], store["r2"], 0.0).sum(axis=1)
    out = np.zeros((len(pos), 4))
    for i, p in enumerate(pos):
        r = zrow(int(p))
        xd, vd = day_xv(store["r2"][r], store["present"][r], rv_all[max(0, r - LOOKBACK):r] if r >= LOOKBACK else rv_all[:0])
        out[i] = (e[i], xd, s[i], vd)
    return out


def std_rows(pool: np.ndarray, n: int) -> np.ndarray:
    """Строки, по которым считаются среднее и σ стандартизации: ТОЛЬКО пул (мутация: все n строк — вся выборка)."""
    return pool


def standardize(Z: np.ndarray, pool: np.ndarray) -> np.ndarray:
    rows = std_rows(pool, len(Z))
    mu, sd = Z[rows].mean(axis=0), Z[rows].std(axis=0)
    return (Z - mu) / np.where(sd > 0, sd, 1.0)


def logit_fit(Zt: np.ndarray, y: np.ndarray, pool: np.ndarray, to: np.ndarray) -> np.ndarray:
    """Вероятность 1[D<0] на строках `to`; в пуле один класс → доля класса (константа)."""
    yy = y[pool]
    if yy.min() == yy.max():
        return np.full(len(to), float(yy.mean()))
    m = LogisticRegression(C=LOGIT_C, solver="lbfgs", max_iter=1000).fit(Zt[pool], yy)
    return m.predict_proba(Zt[to])[:, 1]


def ridge_fit(Zt: np.ndarray, D: np.ndarray, pool: np.ndarray, to: np.ndarray) -> np.ndarray:
    """Гребень: свободный член не штрафуется (центровка), α = 1."""
    X, y = Zt[pool], D[pool]
    xm, ym = X.mean(axis=0), y.mean()
    beta = np.linalg.solve((X - xm).T @ (X - xm) + RIDGE_ALPHA * np.eye(X.shape[1]), (X - xm).T @ (y - ym))
    return ym + (Zt[to] - xm) @ beta


def state_share(e: np.ndarray, D: np.ndarray, pool: np.ndarray, to: np.ndarray) -> np.ndarray:
    """(в): доля D < 0 среди пар пула с тем же e_d; нет таких пар — доля по всему пулу."""
    neg, base = (D[pool] < 0).astype(float), float((D[pool] < 0).mean())
    return np.array([neg[e[pool] == e[i]].mean() if (e[pool] == e[i]).any() else base for i in to])


def walk(Z: np.ndarray, D: np.ndarray, d, tp) -> pd.DataFrame:
    """Помесячно с 2018-07-02 -> по дням d ≥ активации: pa, pb, pc, pg, n_pool, n_e_pool."""
    d, tp, n = pd.DatetimeIndex(d), pd.DatetimeIndex(tp), len(D)
    Zs, y, e = np.roll(Z, LAG, axis=0), (D < 0).astype(int), Z[:, 0].astype(int)
    starts = [p for p in W.month_starts(d) if d[p] >= W.ACTIVATION] + [n]
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        pool, to = W.closed_pool(tp, d, d[a]), np.arange(a, b)
        Zt, Zst = standardize(Z, pool), standardize(Zs, pool)
        rows.append(pd.DataFrame(dict(idx=to, pa=logit_fit(Zt, y, pool, to), pb=ridge_fit(Zt, D, pool, to), pc=state_share(e, D, pool, to),
                                      pg=logit_fit(Zst, y, pool, to), n_pool=len(pool), n_e_pool=int(e[pool].sum()))))
    return pd.concat(rows, ignore_index=True)
