"""Варианты для тестов утечки (О10). Это контрольные варианты, в таблицы прогнозов они не попадают.

Общий контракт для любой модели этапов 2–3: (Xtrain, ytrain, xnew) -> (ln f, s2); варианты меняют только вход.
"""
from __future__ import annotations

import numpy as np

from models_v2 import har
from models_v2.metrics import qlike


def _fit_pred(Xt, yt, xnew):
    beta, s2, _ = har.ols(Xt, yt)
    return float(beta[0] + beta[1:] @ xnew), s2


def work(p, i, cols):
    Xt, yt = har.pairs(p, i, cols)
    return _fit_pred(Xt, yt, p.X[i, list(cols)])


def oracle(p, i, cols):
    """(i) y_{d+1} доступен как признак: в обучении признак = цель пары, на d — ln RV_{d+1}."""
    Xt, yt = har.pairs(p, i, cols)
    return _fit_pred(np.column_stack([Xt, yt]), yt, np.append(p.X[i, list(cols)], p.lnrv[i + 1]))


def full_sample(p, i, cols):
    """(ii) обучение на всей выборке (в т.ч. после i): пары t+1 <= n-1."""
    Xt, yt = har.pairs(p, len(p.rv) - 1, cols)
    return _fit_pred(Xt, yt, p.X[i, list(cols)])


def permuted(p, i, cols, seed=42):
    """(iii) строки X перемешаны внутри окна обучения (связь с y разрушена)."""
    Xt, yt = har.pairs(p, i, cols)
    perm = np.random.default_rng(seed + int(i)).permutation(len(yt))
    return _fit_pred(Xt[perm], yt, p.X[i, list(cols)])


def const_control(p, i, cols):
    """Контроль «ничего не знает» (к1): расширяющееся среднее ln RV по парам окна, s2 = дисперсия цели."""
    _, yt = har.pairs(p, i, cols)
    return float(yt.mean()), float(yt.var(ddof=1))


def qlike_variant(p, grid, cols, fn, bias: bool) -> float:
    rv, s2 = [], []
    for i in grid:
        f, v = fn(p, int(i), cols)
        s2.append(np.exp(f + (v / 2 if bias else 0.0)))
        rv.append(p.rv[i + 1])
    return float(qlike(rv, s2).mean())
