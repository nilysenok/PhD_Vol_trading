"""Обобщённый запуск бустинга на «дизайне»: матрица X, цель y, столбцы cols (Б-1…Б-5, Ж-198).

Пары (x_t, y_t): t+1 <= T, один сегмент, столбцы need и y_t конечны; ES — последние 252 пары, эмбарго 1 (boost.split_pairs).
Модели на диск не сохраняются.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from models_v2.s2 import boost

SEED = 42
LINEAR_LAMBDA = boost.FIXED["l2"]


@dataclass
class Design:
    X: np.ndarray
    seg: np.ndarray
    y: np.ndarray           # y[t] — цель пары (x_t, y_t); NaN, если цели нет
    cols: tuple
    need: tuple = (0, 1, 2, 3)
    linear: bool = False


def pair_index(D: Design, T: int) -> np.ndarray:
    t = np.arange(0, T)
    ok = (D.seg[t] == D.seg[t + 1]) & np.isfinite(D.X[t][:, list(D.need)]).all(axis=1) & np.isfinite(D.y[t])
    return t[ok]


def fit_linear(depth, delta, Xtr, ytr, Xes, yes):
    """LightGBM linear_tree=True (Б-3), потеря MSE; те же замороженные параметры; linear_lambda = L2."""
    p = dict(objective="regression", metric="l2", learning_rate=boost.FIXED["lr"], max_depth=depth,
             num_leaves=2 ** depth, min_data_in_leaf=boost.FIXED["min_leaf"], bagging_fraction=boost.FIXED["subsample"],
             bagging_freq=1, lambda_l2=boost.FIXED["l2"], linear_tree=True, linear_lambda=LINEAR_LAMBDA,
             seed=SEED, deterministic=True, force_row_wise=True, num_threads=1, verbose=-1)
    ds = lgb.Dataset(Xtr, ytr)
    m = lgb.train(p, ds, boost.MAX_TREES, valid_sets=[lgb.Dataset(Xes, yes, reference=ds)],
                  callbacks=[lgb.early_stopping(boost.PATIENCE, verbose=False)])
    k = m.best_iteration or boost.MAX_TREES
    pred = lambda X: m.predict(X, num_iteration=k)  # noqa: E731
    return pred, k, m.feature_importance("gain"), float(np.mean((yes - pred(Xes)) ** 2)), float(np.mean((ytr - pred(Xtr)) ** 2))


def run_design(D: Design, spec: boost.Spec, depth: int, delta: float, T: int, d_idx, variant: str = "work") -> dict:
    """f(x_d) для строк d_idx; variant: work | oracle (цель как признак) | perm (X перемешан в окне) | full (T = всё)."""
    t0, cols, d = time.time(), list(D.cols), np.asarray(d_idx)
    tr, es = boost.split_pairs(pair_index(D, T))
    Xtr, ytr, Xes, yes, Xnew = D.X[tr][:, cols], D.y[tr], D.X[es][:, cols], D.y[es], D.X[d][:, cols]
    if variant == "oracle":
        Xtr, Xes, Xnew = np.column_stack([Xtr, ytr]), np.column_stack([Xes, yes]), np.column_stack([Xnew, D.y[d]])
    elif variant == "perm":
        perm = np.random.default_rng(SEED + int(T)).permutation(len(tr) + len(es))
        Xall = np.vstack([Xtr, Xes])[perm]
        Xtr, Xes = Xall[:len(tr)], Xall[len(tr):]
    if D.linear:
        pred, k, imp, s2_es, _ = fit_linear(depth, delta, Xtr, ytr, Xes, yes)
    else:
        pred, k, imp, s2_es, _ = boost.fit(spec, depth, delta, Xtr, ytr, Xes, yes)
    imp = imp[:len(cols)]
    return {"f": pred(Xnew), "k": k, "s2_es": s2_es, "imp": imp / imp.sum() if imp.sum() > 0 else imp,
            "no_split": bool(imp.sum() == 0), "n_train": len(tr), "sec": time.time() - t0}
