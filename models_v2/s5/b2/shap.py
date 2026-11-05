"""B2: диагностика — доступность окон по годам и важность признаков (TreeSHAP, gain). Не доказательство вклада."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from models_v2.s4 import fit, rows, runner
from models_v2.s5.b2 import engine

WINDOWS = (252, 504, 756, 1008)


def availability(F, grid, dz) -> pd.DataFrame:
    """Для каждого rolling N и года цели: число пар на датах сетки (min/mean) и доля дат с полным окном N."""
    n = np.array([len(runner.pairs_upto(dz, F, int(d))) for d in grid])
    yr = np.array([rows.year_of(F, int(d)) for d in grid])
    rec = []
    for N in WINDOWS:
        for y in sorted(set(yr)):
            m = yr == y
            rec.append({"window": N, "year": int(y), "n_dates": int(m.sum()), "pairs_min": int(n[m].min()), "pairs_mean": float(n[m].mean()),
                        "pairs_used_mean": float(np.minimum(n[m], N).mean()), "share_full_window": float((n[m] >= N).mean())})
    return pd.DataFrame(rec)


def importance(L: dict, cfgs: list, base_names) -> pd.DataFrame:
    F, grid, rec = L["F"], L["grid"], []
    yr = np.array([rows.year_of(F, int(d)) for d in grid])
    for c in cfgs:
        dz = engine.make_dz(L["dz0"], L["feats"], c)
        names = list(base_names) + list(c.add)
        for y in range(2017, 2027):
            T = int(grid[np.flatnonzero(yr == y)[0]])
            hp, idx = L["i2"][y], runner.pairs_upto(dz, F, T)
            k = fit.fit_i1(dz.X, dz.y, idx, hp)["k"]
            bst = xgb.train(fit._params(hp, None, float(np.mean(dz.y[idx]))), xgb.DMatrix(dz.X[idx], label=dz.y[idx]), k)
            sh = np.abs(bst.predict(xgb.DMatrix(dz.X[idx]), pred_contribs=True)[:, :-1]).mean(axis=0)
            g = bst.get_score(importance_type="total_gain")
            tot = sum(g.values()) or 1.0
            for j, nm in enumerate(names):
                rec.append({"model": c.name, "year": y, "T": T, "trees": k, "feature": nm, "mean_abs_shap": float(sh[j]),
                            "gain_share": g.get("f%d" % j, 0.0) / tot, "is_new": nm in c.add})
    return pd.DataFrame(rec)
