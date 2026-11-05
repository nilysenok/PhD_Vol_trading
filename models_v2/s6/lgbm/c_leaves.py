"""Таблица num_leaves × min_data_in_leaf × размер окна (контроль, не trials): доля листьев с наблюдений < 2·min_data, число листьев, QLIKE на прореженной сетке
(каждая 10-я дата сетки, 227 дат). Фиксированные HP: λ2 = 10, lr = 0,05, ff = 0,8, без bagging; признаки — база C-L0; окно = последние N пар (0 — расширяющееся)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6.lgbm import c_hp, c_l0, c_world, fitl, job

NL, MD, WIN, STEP = (7, 15, 31), (30, 100, 200), (504, 1008, 0), 10


def one(F, sp, eps, nl: int, md: int, win: int, d: int) -> dict:
    idx = job.pairs(F, sp, d)
    idx = idx[-win:] if win else idx
    m = fitl.fit_l(sp.X, sp.y, idx, c_hp.HpC(nl, md, 10.0, 0.05, 0.8, 1.0), "ln")
    t = m["fin"].trees_to_dataframe()
    c = t.loc[t["split_feature"].isna(), "count"].to_numpy(float)
    s2, _ = fitl.sigma2("ln", [d], [float(m["predict"](sp.X[[d]])[0])], m["s2_es"], sp.off, np.exp(sp.off), eps)
    return {"q": float(metrics.qlike(np.array([F.rv[d + 1]]), np.asarray(s2))[0]), "n_leaves": len(c), "share_lt_2min": float(np.mean(c < 2 * md)),
            "leaf_min": float(c.min()), "k": m["k"], "n_pairs": len(idx)}


def _job(a):
    nl, md, win, d = a
    return a, one(job._S["F"], job._S["specs"]["leaf"], job._S["eps"], nl, md, win, d)


def summarize(res: dict) -> pd.DataFrame:
    rec = []
    for (nl, md, win), g in pd.DataFrame([{"nl": k[0], "md": k[1], "win": k[2], **v} for k, v in res.items()]).groupby(["nl", "md", "win"]):
        rec.append({"num_leaves": nl, "min_data": md, "window_pairs": win or "expanding", "n_dates": len(g), "qlike_thin": g.q.mean(), "n_leaves_mean": g.n_leaves.mean(),
                    "share_leaves_lt_2min": g.share_lt_2min.mean(), "leaf_min_median": g.leaf_min.median(), "trees_mean": g.k.mean(), "n_pairs_mean": g.n_pairs.mean()})
    df = pd.DataFrame(rec)
    e = df[df.window_pairs == "expanding"].set_index(["num_leaves", "min_data"])["qlike_thin"]
    df["d_qlike_vs_expanding"] = [r.qlike_thin - e[(r.num_leaves, r.min_data)] for r in df.itertuples()]
    return df


def stage(W: dict) -> None:
    _, spec, _ = c_l0.base_cfg(W)
    dates = [int(d) for d in W["grid"][::STEP]]
    args = [(nl, md, win, d) for nl, md, win in itertools.product(NL, MD, WIN) for d in dates]
    with job.pool(W["F"], {"leaf": spec}, W["eps"], 4) as p:
        res = dict(p.imap_unordered(_job, args, chunksize=16))
    df = summarize(res)
    c_world.write(df, c_world.tab.REP / "C_LEAVES_NL_MD_WIN.csv", "num_leaves × min_data × окно: контроль (trials 0), %d дат (каждая %d-я), HP: λ2=10 lr=0,05 ff=0,8" % (len(dates), STEP))
    print(df.to_string(), flush=True)
