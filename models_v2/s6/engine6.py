"""Волна A: LightGBM-контроли на дизайне Б-2 — варианты leaf-wise (S1, S2), искажение цели, статистика листьев.
Ядро fit_var при variant='base', ymod='none' побитно равно `s5/d/lgbfit.fit_lgbx` (тест). Пул spawn <= 5 процессов."""
from __future__ import annotations

import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from models_v2.s4 import fit, rows, runner
from models_v2.s5.d import lgbfit

PROCS = 5
VARIANTS = ("base", "S1", "S2")      # S1: max_depth=-1, num_leaves=2^depth; S2: max_depth=-1, num_leaves=31
YMODS = ("none", "clip99", "drop5")
_W = None


def var_params(hp, seed: int, variant: str = "base") -> dict:
    p = lgbfit.params(hp, seed)
    if variant in ("S1", "S2"):
        p["max_depth"] = -1
    if variant == "S2":
        p["num_leaves"] = 31
    if variant not in VARIANTS:
        raise ValueError(variant)
    return p


def apply_ymod(y: np.ndarray, idx: np.ndarray, ymod: str) -> tuple:
    """-> (y, idx). clip99: цель сверху ограничена 99-м процентилем обучающих целей; drop5: без 5 наибольших целей."""
    if ymod == "clip99":
        return np.minimum(y, np.quantile(y[idx], 0.99)), idx
    if ymod == "drop5":
        return y, np.sort(idx[np.argsort(y[idx])[:-5]])
    if ymod != "none":
        raise ValueError(ymod)
    return y, idx


def fit_var(X, y, idx, hp, seed: int, variant: str = "base", ymod: str = "none") -> dict:
    """Как fit_lgbx (без весов): I1 — ранняя остановка по последним 252 парам, финал на всех парах с k деревьями."""
    y, idx = apply_ymod(y, idx, ymod)
    tr, es, _ = fit.split(idx)
    p = var_params(hp, seed, variant)
    dtr = lgb.Dataset(X[tr], y[tr])
    b = lgb.train(p, dtr, hp.max_trees, valid_sets=[lgb.Dataset(X[es], y[es], reference=dtr)],
                  callbacks=[lgb.early_stopping(fit.PATIENCE, verbose=False)])
    k = max(1, b.best_iteration)
    s2 = float(np.mean((y[es] - b.predict(X[es], num_iteration=k)) ** 2))
    fin = lgb.train(p, lgb.Dataset(X[idx], y[idx]), k)
    return {"fin": fin, "k": k, "s2_es": s2, "predict": lambda Z: np.asarray(fin.predict(np.atleast_2d(Z)), float)}


def leaf_stats(fin, min_data: int) -> dict:
    """Наблюдений в листьях финальной модели (count листьев по всем деревьям)."""
    t = fin.trees_to_dataframe()
    c = t.loc[t["split_feature"].isna(), "count"].to_numpy(float)
    return {"n_leaves": len(c), "leaf_min": float(c.min()), "leaf_median": float(np.median(c)), "share_lt_2min": float(np.mean(c < 2 * min_data))}


def _init(w):
    global _W
    _W = w


def _job(a):
    d, hp, seed, variant, ymod, leaves = a
    F, dz = _W.F, _W.dzs["b2"]
    t0 = time.time()
    m = fit_var(dz.X, dz.y, runner.pairs_upto(dz, F, d), hp, seed, variant, ymod)
    f = float(m["predict"](dz.X[[d]])[0])
    out = {"f": f, "k": m["k"], "s2_es": m["s2_es"], "secs": time.time() - t0}
    if leaves:
        out.update(leaf_stats(m["fin"], max(1, int(hp.min_child))))
    return d, out


def pool(w, procs: int = PROCS):
    return mp.get_context("spawn").Pool(min(procs, PROCS), initializer=_init, initargs=(w,))


def run(p, w, hp_y: dict, dates, seed: int = 42, variant: str = "base", ymod: str = "none", leaves: bool = False) -> pd.DataFrame:
    """Строка на каждую d из dates: f (ln-прогноз), k, s2_es, sigma2 = exp(off + f + s2/2), target_date, rv."""
    F, dz = w.F, w.dzs["b2"]
    res = dict(p.imap_unordered(_job, [(int(d), hp_y[rows.year_of(F, int(d))], seed, variant, ymod, leaves) for d in dates], chunksize=4))
    i = np.asarray(dates, int)
    df = pd.DataFrame([res[int(d)] for d in i])
    df.insert(0, "d", F.dates[i])
    df.insert(1, "target_date", F.dates[i + 1])
    df["sigma2"] = np.exp(dz.off[i] + df["f"].to_numpy() + df["s2_es"].to_numpy() / 2)
    df["rv"] = F.rv[i + 1]
    return df
