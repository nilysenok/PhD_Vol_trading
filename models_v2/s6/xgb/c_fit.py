"""Обучение волны C: схема И1 (`s4/fit.fit_i1`) с осями gamma / reg_alpha / max_delta_step и seed для двух потерь: 'l2' (ln-цель, s² блока остановки)
и 'gamma' (reg:gamma, log-link, f = ln μ). При нулевом сдвиге и seed 42 побитно равна `fit.fit_i1` и `fitobj._xgb` (тесты C2, гейт C1)."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from models_v2.s4 import fit
from models_v2.s5.c import fitx


def params(hp, base: float, seed: int, loss: str) -> dict:
    p = fitx.params(hp, None, base, seed)
    mds = float(getattr(hp, "mds", 0.0))
    if mds > 0:
        p["max_delta_step"] = mds
    if loss == "gamma":
        p.update(objective="reg:gamma", eval_metric="gamma-nloglik")
    return p


def _wm(v) -> float:
    return float(np.mean(v))


def fit_c(X, y, idx, hp, loss: str = "l2", seed: int = fit.SEED) -> dict:
    """-> predict (f: l2 — ln-отклонение, gamma — ln μ), k (деревьев), s2_es (l2: средний квадрат остатка блока остановки; gamma: nan), n_train."""
    tr, es, shrunk = fit.split(idx)
    dtr, des = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X[es], label=y[es])
    b = xgb.train(params(hp, _wm(y[tr]), seed, loss), dtr, hp.max_trees, evals=[(des, "es")], early_stopping_rounds=fit.PATIENCE, verbose_eval=False)
    k = b.best_iteration + 1
    fin = xgb.train(params(hp, _wm(y[idx]), seed, loss), xgb.DMatrix(X[idx], label=y[idx]), k)
    s2 = float("nan") if loss == "gamma" else _wm((y[es] - b.predict(des, iteration_range=(0, k))) ** 2)

    def predict(Z):
        p = fin.predict(xgb.DMatrix(np.atleast_2d(Z))).astype(float)
        return np.log(p) if loss == "gamma" else p
    return {"predict": predict, "k": k, "s2_es": s2, "n_train": len(idx), "shrunk": shrunk}
