"""Y-B, помесячная переоценка Y3/Y4 (план раздел 8): как Y-A (`x1_ya_walk`: те же закрытые пары, дни переоценки, включение с 2018-07-02 при ≥ 120 закрытых парах, до этого F = X1 побитно, active = 0).
Y3: на каждой переоценке 5 независимых обучений (seeds 0–4: выбор λ + переобучение со стартом γ ~ N(0; 0,1²)); итог F = exp(среднее ln F_s) = X1·exp(среднее g_s). Y4: одно обучение θ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_yb_fit as FT
from models_v2.s16.x1_ya_walk import ACTIVATION, MIN_PAIRS, check_activation, closed_pool, month_starts  # noqa: F401

OUT_COLS = ["d", "t_plus", "rv", "X1", "F", "g", "active", "lam", "theta", "g1", "g2", "g3", "g4"]


def walk(x1: pd.DataFrame, D: dict, mode: str, seeds=FT.SEEDS) -> pd.DataFrame:
    """x1: d, t_plus, rv, F (=X1); D — данные пар `x1_yb_feat.pair_data`. -> прогнозы (OUT_COLS)."""
    d, tp = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"])
    X1, n = x1["F"].to_numpy(float), len(d)
    check_activation(d, tp)
    g, active = np.zeros(n), np.zeros(n)
    par = np.full((n, 6), np.nan)          # λ, θ, γ₁…γ₄ (среднее по seeds)
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = closed_pool(tp, d, d[a])
        fits = [FT.refit(pool, D, mode, s) for s in (seeds if mode == "Y3" else seeds[:1])]
        Db = FT.sub(D, np.arange(a, b))
        g[a:b] = np.mean([x[0] * FT.h_of(x, Db, mode) for _, x in fits], axis=0)
        active[a:b] = 1.0
        par[a:b, 0] = np.mean([lam for lam, _ in fits])
        for j in range(len(fits[0][1])):
            par[a:b, 1 + j] = np.mean([x[j] for _, x in fits])
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=x1["rv"].to_numpy(float), X1=X1, F=X1 * np.exp(g), g=g, active=active.astype(int), lam=par[:, 0], theta=par[:, 1],
                             **{"g%d" % k: par[:, 1 + k] for k in range(1, 5)}))[OUT_COLS]
