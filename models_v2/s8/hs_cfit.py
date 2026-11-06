"""Ж-212 (HS6, G9): residual QLIKE. f = HAR·exp(g), L = ln f + RV/f; по m = ln f: grad = 1 − RV/f, hess = RV/f (> 0). Штраф μ на древесную часть g_tree = m − (lb + c0):
grad += μ·g_tree, hess += μ. offset = lb + c0 (PIT HAR-J(б), c0 = ln среднего RV/HAR по парам окна). HP фиксированы заранее: глубина 1, 150 деревьев, lr 0,03, subsample 0,8,
colsample 0,8, min_child_weight 50, lambda_l2 10. Ежедневное переобучение, расширяющееся окно, пары t+1 <= d, seed 42. g = c0 + g_tree; σ̂² = HAR·exp(λ·g) (λ = 0 ≡ HAR побитно)."""
from __future__ import annotations

import multiprocessing as mp

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from models_v2.s4 import fit

HP = {"trees": 150, "lr": 0.03, "subsample": 0.8, "colsample": 0.8, "mcw": 50, "l2": 10.0}
PROCS = 4
_S: dict = {}


def gh(y, m, off, mu: float = 0.0) -> tuple:
    """Градиент и гессиан по m = ln f для L = m + y·e^{−m} + (μ/2)(m − off)²; off = lb + c0."""
    r = np.asarray(y, float) * np.exp(-np.asarray(m, float))
    return 1.0 - r + mu * (np.asarray(m, float) - off), r + mu


def loss(y, m, off, mu: float = 0.0):
    m = np.asarray(m, float)
    return m + np.asarray(y, float) * np.exp(-m) + 0.5 * mu * (m - off) ** 2


def fit_xgb(X, y, off, mu: float, mono, seed: int):
    p = {"max_depth": 1, "eta": HP["lr"], "subsample": HP["subsample"], "colsample_bytree": HP["colsample"], "min_child_weight": HP["mcw"], "reg_lambda": HP["l2"],
         "tree_method": "hist", "nthread": 1, "seed": seed, "base_score": 0.0, "disable_default_eval_metric": 1}
    if any(mono):
        p["monotone_constraints"] = "(%s)" % ",".join(str(int(v)) for v in mono)
    d = xgb.DMatrix(X, label=y, base_margin=off)
    b = xgb.train(p, d, HP["trees"], obj=lambda preds, dm: gh(dm.get_label(), preds, off, mu))
    return lambda Z: b.predict(xgb.DMatrix(np.atleast_2d(Z), base_margin=np.zeros(len(np.atleast_2d(Z)))), output_margin=True).astype(float)


def fit_lgbm(X, y, off, mu: float, mono, seed: int):
    p = {"num_leaves": 2, "max_depth": 1, "learning_rate": HP["lr"], "bagging_fraction": HP["subsample"], "bagging_freq": 1, "feature_fraction": HP["colsample"],
         "min_data_in_leaf": HP["mcw"], "lambda_l2": HP["l2"], "seed": seed, "num_threads": 1, "deterministic": True, "force_row_wise": True, "verbose": -1,
         "objective": lambda preds, ds: gh(ds.get_label(), preds, off, mu)}
    if any(mono):
        p["monotone_constraints"] = [int(v) for v in mono]
    b = lgb.train(p, lgb.Dataset(X, label=y, init_score=off), HP["trees"])
    return lambda Z: np.asarray(b.predict(np.atleast_2d(Z), raw_score=True), float)


FITS = {"XGB": fit_xgb, "LGBM": fit_lgbm}


def fit_day(X, y, lb, idx, d: int, lib: str, mu: float, mono, seed: int = 42) -> tuple:
    """-> (g_tree(X_d), c0) по парам idx (t+1 <= d); offset = lb + c0."""
    c0 = float(np.log(np.mean(y[idx] / np.exp(lb[idx]))))
    pr = FITS[lib](X[idx], y[idx], lb[idx] + c0, mu, mono, seed)
    return float(pr(X[[d]])[0]), c0


def init(seg, gate, X, y, lb, lib, mu, mono):
    _S.update(seg=seg, gate=gate, X=X, y=y, lb=lb, lib=lib, mu=mu, mono=mono)


def day_job(d):
    S = _S
    idx = fit.pair_index(S["seg"], S["gate"], S["y"], (0, 1, 2, 3), d)
    return d, fit_day(S["X"], S["y"], S["lb"], idx, d, S["lib"], S["mu"], S["mono"])


def run(F, gate, X, lb, eps, grid, lib: str, mu: float = 0.0, mono=(), procs: int = PROCS) -> np.ndarray:
    """-> g (c0 + g_tree) на сетке: ln(σ̂²_C / HAR). y_t = max(RV_{t+1}, ε_t); строки без lb исключаются из пар."""
    n = len(F.rv)
    rvn = np.full(n, np.nan)
    rvn[:-1] = F.rv[1:]
    y = np.where(np.isfinite(lb), np.maximum(rvn, eps), np.nan)
    mono = tuple(mono) if len(mono) else (0,) * X.shape[1]
    with mp.get_context("spawn").Pool(min(procs, PROCS), initializer=init, initargs=(F.seg, gate, X, y, lb, lib, mu, mono)) as p:
        res = dict(p.imap_unordered(day_job, [int(d) for d in grid], chunksize=8))
    return np.array([res[int(d)][1] + res[int(d)][0] for d in grid])
