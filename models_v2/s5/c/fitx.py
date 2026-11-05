"""C: копия `s4/fit.fit_i1` (И1) с двумя дополнительными осями XGBoost (gamma, reg_alpha) и параметром seed.
При gamma = alpha = 0 и seed 42 побитно равна `fit.fit_i1` (гейт G2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from models_v2.s4 import fit


@dataclass(frozen=True)
class HPX(fit.HP):
    gamma: float = 0.0
    alpha: float = 0.0


def params(hp, mono, base: float, seed: int) -> dict:
    p = fit._params(hp, mono, base)
    p.update(gamma=float(getattr(hp, "gamma", 0.0)), reg_alpha=float(getattr(hp, "alpha", 0.0)), seed=int(seed))
    return p


def fit_i1x(X, y, idx, hp, w=None, mono=None, seed: int = fit.SEED, max_trees: int | None = None) -> dict:
    max_trees = hp.max_trees if max_trees is None else max_trees
    tr, es, shrunk = fit.split(idx)
    g = (lambda a: None if w is None else w[a])
    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=g(tr))
    des = xgb.DMatrix(X[es], label=y[es], weight=g(es))
    b = xgb.train(params(hp, mono, fit._wmean(y[tr], g(tr)), seed), dtr, max_trees, evals=[(des, "es")],
                  early_stopping_rounds=fit.PATIENCE, verbose_eval=False)
    k = b.best_iteration + 1
    r_es = y[es] - b.predict(des, iteration_range=(0, k))
    r_tr = y[tr] - b.predict(dtr, iteration_range=(0, k))
    fin = xgb.train(params(hp, mono, fit._wmean(y[idx], g(idx)), seed), xgb.DMatrix(X[idx], label=y[idx], weight=g(idx)), k)
    s2_tr = fit._wmean((y[idx] - fin.predict(xgb.DMatrix(X[idx]))) ** 2, g(idx))
    return {"predict": lambda Z: fin.predict(xgb.DMatrix(np.atleast_2d(Z))).astype(float), "k": k,
            "s2_es": fit._wmean(r_es ** 2, g(es)), "s2_tr_prelim": fit._wmean(r_tr ** 2, g(tr)), "s2_tr": s2_tr,
            "n_train": len(idx), "shrunk": shrunk, "train_idx": idx, "es_idx": es}
