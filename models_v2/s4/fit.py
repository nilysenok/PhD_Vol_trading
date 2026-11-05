"""И1 (Ж-200): ранняя остановка без потери данных.

n* — по последним 252 парам окна (эмбарго 1, терпение 100, потолок 3000); финальная модель — на ВСЕХ парах окна
с n* деревьями. s² для (б) — средний (взвешенный) квадрат остатка предварительной модели на блоке остановки.
Пары — индексы строк t (обучающая пара (x_t, y_t), цель y_t = y[t]); окно: t+1 <= T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

ES_N, EMBARGO, PATIENCE, MAX_TREES, MIN_TRAIN, MIN_ES = 252, 1, 100, 3000, 100, 30
SEED = 42
HALF_LIFE = 504   # годных дня (2 года)


@dataclass(frozen=True)
class HP:
    depth: int = 3
    min_child: float = 50.0
    l2: float = 5.0
    lr: float = 0.05
    subsample: float = 0.8
    colsample: float = 1.0
    max_trees: int = MAX_TREES


def pair_index(seg: np.ndarray, X: np.ndarray, y: np.ndarray, need, T: int) -> np.ndarray:
    """Строки t с t+1 <= T, одним сегментом, конечными need-признаками и целью."""
    t = np.arange(0, T)
    ok = (seg[t] == seg[t + 1]) & np.isfinite(X[t][:, list(need)]).all(axis=1) & np.isfinite(y[t])
    return t[ok]


def split(idx: np.ndarray) -> tuple:
    """(train, es, shrunk): ES — последние 252 пары, эмбарго 1; при нехватке пар блок сокращается до трети (>= 30)."""
    n, es_n, shrunk = len(idx), ES_N, False
    if n < ES_N + EMBARGO + MIN_TRAIN:
        es_n, shrunk = max(MIN_ES, (n - EMBARGO) // 3), True
    if n - EMBARGO - es_n < MIN_ES:
        raise ValueError("мало пар: %d" % n)
    return idx[:n - EMBARGO - es_n], idx[n - es_n:], shrunk


def weights(idx: np.ndarray, T: int, half_life: float = HALF_LIFE) -> np.ndarray:
    """w = 0,5^((T − t)/half_life): возраст — строки между t и концом окна T; свежее тяжелее."""
    return 0.5 ** ((T - idx) / half_life)


def _params(hp: HP, mono, base: float) -> dict:
    p = dict(objective="reg:squarederror", eval_metric="rmse", eta=hp.lr, max_depth=hp.depth,
             min_child_weight=hp.min_child, subsample=hp.subsample, colsample_bytree=hp.colsample, reg_lambda=hp.l2, tree_method="hist",
             nthread=1, seed=SEED, verbosity=0, base_score=base)
    if mono is not None and any(mono):
        p["monotone_constraints"] = "(" + ",".join(str(int(v)) for v in mono) + ")"
    return p


def _wmean(v, w) -> float:
    return float(np.average(v, weights=w)) if w is not None else float(np.mean(v))


def fit_i1(X, y, idx, hp: HP, w=None, mono=None, max_trees: int | None = None) -> dict:
    """X, y — полные массивы по строкам; idx — пары окна; w — веса по строкам t (или None)."""
    max_trees = hp.max_trees if max_trees is None else max_trees
    tr, es, shrunk = split(idx)
    g = (lambda a: None if w is None else w[a])
    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=g(tr))
    des = xgb.DMatrix(X[es], label=y[es], weight=g(es))
    b = xgb.train(_params(hp, mono, _wmean(y[tr], g(tr))), dtr, max_trees, evals=[(des, "es")],
                  early_stopping_rounds=PATIENCE, verbose_eval=False)
    k = b.best_iteration + 1
    r_es = y[es] - b.predict(des, iteration_range=(0, k))
    r_tr = y[tr] - b.predict(dtr, iteration_range=(0, k))
    fin = xgb.train(_params(hp, mono, _wmean(y[idx], g(idx))), xgb.DMatrix(X[idx], label=y[idx], weight=g(idx)), k)
    s2_tr = _wmean((y[idx] - fin.predict(xgb.DMatrix(X[idx]))) ** 2, g(idx))
    return {"predict": lambda Z: fin.predict(xgb.DMatrix(np.atleast_2d(Z))).astype(float), "k": k, "s2_es": _wmean(r_es ** 2, g(es)),
            "s2_tr_prelim": _wmean(r_tr ** 2, g(tr)), "s2_tr": s2_tr, "n_train": len(idx), "shrunk": shrunk,
            "train_idx": idx, "es_idx": es}
