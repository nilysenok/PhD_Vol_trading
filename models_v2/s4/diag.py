"""Диагностика Ж-200: (а) недоучка по годам, (б) тест переобучения на всей выборке."""
from __future__ import annotations

import multiprocessing as mp

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from models_v2 import har, metrics, prov
from models_v2.s2 import boost
from models_v2.s2 import features as ft
from models_v2.s3 import gen, resid, scale

OUT = prov.ROOT / "data/regen/v2_sber_h1"
_G = {}


def p2_params() -> tuple:
    t = pd.read_csv(prov.ROOT / "docs/tables/v2_sber_h1/p2_tuning.csv", comment="#")
    r = t[(t.lib == "xgb") & (t["loss"] == "mse") & t.chosen].iloc[0]
    return int(r.depth), float(r.delta)


def design_b2(F):
    S = scale.scale(F, "m")
    D = gen.Design(scale.normalize(F.X, S), F.seg, resid.target_minus(F, np.log(S)), tuple(range(ft.N_BASE)))
    return D, np.log(S)


def _init(F, D, depth, delta):
    _G.update(F=F, D=D, depth=depth, delta=delta)


def _b2_job(T):
    D, spec = _G["D"], boost.Spec("xgb", "mse", False, False)
    tr, es = boost.split_pairs(gen.pair_index(D, T))
    cols = list(D.cols)
    pred, k, _, s2_es, s2_tr = boost.fit(spec, _G["depth"], _G["delta"], D.X[tr][:, cols], D.y[tr], D.X[es][:, cols], D.y[es])
    return T, float(pred(D.X[[T]][:, cols])[0]), k, s2_es, s2_tr


def _share(F, d_rows, sigma2) -> np.ndarray:
    ls = np.log(sigma2)
    return np.abs(ls - F.X[d_rows, 2]) < np.abs(ls - F.X[d_rows, 0])


def _agg(F, model, d_rows, k, s2_tr, s2_es, sigma2) -> list:
    yr = pd.Series(F.dates[d_rows + 1]).dt.year.to_numpy()
    close = _share(F, d_rows, sigma2)
    rows = []
    for y in sorted(set(yr)):
        m = yr == y
        row = {"model": model, "year": int(y), "n_dates": int(m.sum()), "share_closer_rv_m": float(close[m].mean())}
        if k is not None:
            row.update(trees_mean=float(np.mean(k[m])), trees_min=int(np.min(k[m])), trees_max=int(np.max(k[m])),
                       mse_train=float(np.mean(s2_tr[m])), mse_es=float(np.mean(s2_es[m])))
        rows.append(row)
    return rows


def diag_a(F, grid, procs: int = 4):
    rt = pd.read_parquet(OUT / "retrain_p2.parquet")
    f2, f1, fb = (pd.read_parquet(OUT / ("forecasts_%s.parquet" % n)) for n in ("p2", "p1", "pb"))
    ix = lambda d: np.searchsorted(F.dates, d.to_numpy(dtype="datetime64[ns]"))  # noqa: E731
    rows = []
    for lib in ("xgb", "lgbm"):
        m = rt[rt.model == lib + "_mse_nohar_mono"].sort_values("T")
        s = f2[f2.model == lib + "_mse_nohar_mono_b"].sort_values("d")
        T = m["T"].to_numpy()
        assert np.array_equal(T, ix(s["d"])), "T retrain != d прогноза"
        rows += _agg(F, lib + "_mse_nohar_mono_b", T, m.n_trees.to_numpy(), m.s2_tr.to_numpy(), m.s2_es.to_numpy(), s.sigma2.to_numpy())
    D, off = design_b2(F)
    depth, delta = p2_params()
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(F, D, depth, delta)) as p:
        res = sorted(p.imap_unordered(_b2_job, [int(d) for d in grid], chunksize=4))
    T, f, k, s2e, s2t = (np.array(c) for c in zip(*res))
    sig = np.exp(off[T.astype(int)] + f + s2e / 2)
    ref = fb[fb.model == "b2_m_xgb_b__daily"].sort_values("d")
    diff = float(np.max(np.abs(np.log(sig) - np.log(ref.sigma2.to_numpy()))))
    rows += _agg(F, "b2_m_xgb_b", T.astype(int), k, s2t, s2e, sig)
    h = f1[f1.model == "harj_b"].sort_values("d")
    rows += _agg(F, "harj_b (ориентир)", ix(h["d"]), None, None, None, h.sigma2.to_numpy())
    return pd.DataFrame(rows), diff


def _lgb(X, y, mono, depth):
    p = dict(objective="regression", learning_rate=0.03, max_depth=depth, num_leaves=2 ** depth, min_data_in_leaf=10,
             lambda_l2=0.0, bagging_fraction=1.0, seed=42, deterministic=True, force_row_wise=True, num_threads=1, verbose=-1)
    if mono is not None:
        p["monotone_constraints"] = list(mono)
    m = lgb.train(p, lgb.Dataset(X, y), 3000)
    return lambda Z: m.predict(Z)


def _xgb(X, y, mono, depth):
    p = dict(objective="reg:squarederror", eta=0.03, max_depth=depth, min_child_weight=10, subsample=1.0, reg_lambda=0.0,
             tree_method="hist", nthread=1, seed=42, verbosity=0, base_score=float(y.mean()))
    if mono is not None:
        p["monotone_constraints"] = "(" + ",".join(str(v) for v in mono) + ")"
    b = xgb.train(p, xgb.DMatrix(X, label=y), 3000)
    return lambda Z: b.predict(xgb.DMatrix(Z))


def _b_job(a):
    name, lib, mono_on, design = a
    F, D, off = _G["F"], _G["D"][design], _G["off"][design]
    T = len(F.dates) - 1
    idx = gen.pair_index(D, T)
    cols = list(D.cols)
    mono = [1 if (mono_on and j in ft.MONO) else 0 for j in range(len(cols))] if mono_on else None
    pred = (_xgb if lib == "xgb" else _lgb)(D.X[idx][:, cols], D.y[idx], mono, 6)
    s2_tr = float(np.mean((D.y[idx] - pred(D.X[idx][:, cols])) ** 2))
    g = _G["grid"]
    f = pred(D.X[g][:, cols])
    out = {}
    for kind, add in (("a", 0.0), ("b", s2_tr / 2)):
        out[kind] = float(metrics.qlike(F.rv[g + 1], np.exp(off[g] + f + add)).mean())
    return {"model": name, "lib": lib, "mono": mono_on, "design": design, "qlike_a": out["a"], "qlike_b": out["b"],
            "n_pairs": len(idx), "s2_train": s2_tr}


def _initb(F, D, off, grid):
    _G.update(F=F, D=D, off=off, grid=grid)


def diag_b(F, grid, procs: int = 4):
    Db, off2 = design_b2(F)
    D0 = gen.Design(F.X[:, :ft.N_BASE], F.seg, resid.target_minus(F, np.zeros(len(F.rv))), tuple(range(ft.N_BASE)))
    jobs = [("base_%s_%s" % (lib, "mono" if mo else "free"), lib, mo, "base") for lib in ("xgb", "lgbm") for mo in (False, True)]
    jobs += [("b2_%s_free" % lib, lib, False, "b2") for lib in ("xgb", "lgbm")]
    with mp.get_context("spawn").Pool(procs, initializer=_initb, initargs=(F, {"base": D0, "b2": Db},
                                      {"base": np.zeros(len(F.rv)), "b2": off2}, grid)) as p:
        rows = p.map(_b_job, jobs)
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    beta, s2, _ = har.ols(*har.pairs(pan, len(F.dates) - 1, har.COLS_HARJ))
    fh = beta[0] + (pan.X[grid] * beta[1:]).sum(axis=1)
    ref = {"harj_a_in_sample": float(metrics.qlike(F.rv[grid + 1], np.exp(fh)).mean()),
           "harj_b_in_sample": float(metrics.qlike(F.rv[grid + 1], np.exp(fh + s2 / 2)).mean()),
           "harj_b_honest": float(metrics.qlike(p1[p1.model == "harj_b"]["rv"], p1[p1.model == "harj_b"]["sigma2"]).mean())}
    df = pd.DataFrame(rows).assign(**ref)
    df["better_than_harj_in_sample"] = df.qlike_b < df.harj_b_in_sample
    return df
