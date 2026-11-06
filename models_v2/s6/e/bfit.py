"""Тип B (R6, G6): residual-гибрид RV_hyb = HAR·exp(g(X)). Объектив QLIKE (loss = ln σ̂² + RV/σ̂²) с offset ln HAR (lb — PIT HAR-J(б)):
XGBoost reg:gamma + base_margin, LightGBM objective=gamma + init_score; градиент 1 − RV·e^{−m}, гессиан RV·e^{−m} (m = ln σ̂²). Аддитивный вариант 'xa': max(HAR + g, ε_PIT).
Константа c0 = ln среднего отношения RV/HAR по обучающим парам входит в offset (аналог base_score): offset = lb + c0. Ежедневное переобучение, расширяющееся окно, схема И1 (`s4/fit`)."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from models_v2.s4 import fit, rows
from models_v2.s5.c import fitx
from models_v2.s6.e import pe
from models_v2.s6.xgb import c_fit

KINDS = ("xq", "lq", "xa")
_S: dict = {}


def qlike_gh(y, m):
    """Градиент и гессиан QLIKE по m = ln σ̂²: loss = m + y·e^{−m}."""
    r = np.asarray(y, float) * np.exp(-np.asarray(m, float))
    return 1.0 - r, r


def c0_of(y, lb) -> float:
    return float(np.log(np.mean(np.asarray(y) / np.exp(lb))))


def fit_xq(X, lb, y, idx, hp, seed: int) -> dict:
    tr, es, _ = fit.split(idx)
    c0 = c0_of(y[tr], lb[tr])
    p = c_fit.params(hp, 1.0, seed, "gamma")
    b = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr], base_margin=lb[tr] + c0), hp.max_trees, evals=[(xgb.DMatrix(X[es], label=y[es], base_margin=lb[es] + c0), "es")],
                  early_stopping_rounds=fit.PATIENCE, verbose_eval=False)
    k, c1 = b.best_iteration + 1, c0_of(y[idx], lb[idx])
    fin = xgb.train(p, xgb.DMatrix(X[idx], label=y[idx], base_margin=lb[idx] + c1), k)
    return {"k": k, "predict": lambda Z, lz: np.log(fin.predict(xgb.DMatrix(np.atleast_2d(Z), base_margin=np.atleast_1d(lz) + c1)).astype(float))}


def fit_lq(X, lb, y, idx, hp, seed: int) -> dict:
    tr, es, _ = fit.split(idx)
    c0 = c0_of(y[tr], lb[tr])
    p = dict(hp.params(seed), objective="gamma", metric="gamma")
    dtr = lgb.Dataset(X[tr], y[tr], init_score=lb[tr] + c0)
    des = lgb.Dataset(X[es], y[es], init_score=lb[es] + c0, reference=dtr)
    b = lgb.train(p, dtr, hp.max_trees, valid_sets=[des], callbacks=[lgb.early_stopping(fit.PATIENCE, verbose=False)])
    k, c1 = max(1, b.best_iteration), c0_of(y[idx], lb[idx])
    fin = lgb.train(p, lgb.Dataset(X[idx], y[idx], init_score=lb[idx] + c1), k)
    return {"k": k, "predict": lambda Z, lz: np.asarray(fin.predict(np.atleast_2d(Z), raw_score=True), float) + np.atleast_1d(lz) + c1}


def day_job(d):
    """-> (d, {'lnmu' (xq/lq) | 'g' (xa), 'k'}) по парам с t+1 <= d (`fit.pair_index`)."""
    S = _S
    idx = fit.pair_index(S["F"].seg, S["X"], S["ypair"], (0, 1, 2, 3), d)
    hp = S["hpy"][rows.year_of(S["F"], d)]
    if S["kind"] == "xa":
        m = fitx.fit_i1x(S["X"], S["ydiff"], idx, hp, seed=S["seed"])
        return d, {"g": float(m["predict"](S["X"][[d]])[0]), "k": m["k"]}
    m = (fit_xq if S["kind"] == "xq" else fit_lq)(S["X"], S["lb"], S["y"], idx, hp, S["seed"])
    return d, {"lnmu": float(m["predict"](S["X"][[d]], S["lb"][[d]])[0]), "k": m["k"]}


def init(F, X, lb, y, ypair, ydiff, hpy, kind, seed):
    _S.update(F=F, X=X, lb=lb, y=y, ypair=ypair, ydiff=ydiff, hpy=hpy, kind=kind, seed=seed)


def run(W: dict, kind: str, X: np.ndarray, hpy: dict, seed: int = 42, procs: int = 4) -> pd.DataFrame:
    """Ряд σ̂² на сетке W['grid'] (d, target_date, rv, sigma2, g, k). y — RV_{t+1}, не меньше PIT-границы ε_t; lb = W['dz'].off."""
    F, g, dz = W["F"], W["grid"], W["dz"]
    n, lb = len(F.rv), dz.off
    rvn = np.full(n, np.nan)
    rvn[:-1] = F.rv[1:]
    y = np.where(np.isfinite(dz.y), np.maximum(rvn, W["eps"]), np.nan)
    ydiff = np.where(np.isfinite(dz.y), rvn - np.exp(lb), np.nan)
    ypair = ydiff if kind == "xa" else dz.y
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=init, initargs=(F, X, lb, y, ypair, ydiff, hpy, kind, seed)) as p:
        res = dict(p.imap_unordered(day_job, [int(d) for d in g], chunksize=8))
    r = [res[int(d)] for d in g]
    if kind == "xa":
        gg = np.array([x["g"] for x in r])
        s2 = np.maximum(np.exp(lb[g]) + gg, W["eps"][g])
    else:
        s2 = np.exp(np.array([x["lnmu"] for x in r]))
        gg = np.log(s2) - lb[g]
    return pd.DataFrame({"d": F.dates[g], "target_date": F.dates[g + 1], "rv": F.rv[g + 1], "sigma2": s2, "g": gg, "k": [x["k"] for x in r], "secs": time.time() - t0})


def key(kind: str, X, seed: int, tag: str = "") -> str:
    h = hashlib.sha256()
    h.update(open(__file__, "rb").read() + kind.encode() + str(seed).encode() + tag.encode() + np.ascontiguousarray(np.nan_to_num(X)).tobytes())
    return h.hexdigest()[:16]


def cached(name: str, k: str, build) -> pd.DataFrame:
    p, kp = pe.OUT / ("b_%s.parquet" % name), pe.OUT / ("b_%s.key" % name)
    if p.exists() and kp.exists() and kp.read_text() == k:
        return pd.read_parquet(p)
    df = build()
    df.to_parquet(p)
    kp.write_text(k)
    return df
