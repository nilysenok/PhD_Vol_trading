"""W9 ПОСЛЕ ПРОСМОТРА, продолжение (11.11, после PH): неопределённость PH-R, чувствительность к длине пула (126, 504) и сочетание с β W5-G.
Только показ; кандидатом ничего не объявляется. Строки -> блок PH2 W9_DIAG (через `x1_w9_diag.py`).
Перенос из scratchpad `w9/posthoc2.py` без изменения логики (лист кодера 11.11.2026)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_t0 as T0
from models_v2.s16 import x1_w9_diag_core as K
from models_v2.s16 import x1_w9_diag_ph as PH

ST, EVX = T0.ST, T0.EVX


def walk_len(x, L: int):
    """PH-R с пулом последних L закрытых пар."""
    rv, lnM = x["rv"].to_numpy(float), np.log(x["M"].to_numpy(float))
    p1, p2 = (x["pi"] * x["DH"]).to_numpy(float), (x["pi"] * x["DG"]).to_numpy(float)
    d, tp = x["d"].to_numpy(), x["t_plus"].to_numpy()
    mon = x["d"].dt.to_period("M").to_numpy()
    lnF = lnM + p1 + p2
    starts = [i for i in range(len(x)) if i == 0 or mon[i] != mon[i - 1]]
    for k, i0 in enumerate(starts):
        i1 = starts[k + 1] if k + 1 < len(starts) else len(x)
        closed = np.where(tp <= d[i0])[0][-L:]
        if len(closed) < PH.MIN_POOL:
            continue
        m = np.zeros(len(x), bool)
        m[closed] = True
        kh, kg = K.fit_kappa2(rv, lnM, p1, p2, m)
        lnF[i0:i1] = lnM[i0:i1] + kh * p1[i0:i1] + kg * p2[i0:i1]
    return np.exp(lnF)


def run(x) -> list:  # xtask: длинная — перенос без изменения логики, один блок PH2
    rv = x["rv"].to_numpy(float)
    X1 = x["F"].to_numpy(float)
    w5 = pd.read_parquet(K.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    w5["d"] = pd.to_datetime(w5["d"])
    w5 = x[["d"]].merge(w5, on="d", how="left")
    if not np.array_equal(w5["X1"].to_numpy(float), X1):
        raise SystemExit("w5g.X1 ≠ x1.F")
    W5 = w5["F"].to_numpy(float)
    corr = np.log(W5 / X1)                       # gap·β W5-G
    PHR, _ = PH.walk(x, "R", True)
    ser = {"PH-R": PHR, "PH-R126": walk_len(x, 126), "PH-R504": walk_len(x, 504), "PH-R+W5G": PHR * np.exp(corr)}
    per = PH.periods(x)
    rows = []
    for name, F in ser.items():
        l = K.qlike(rv, F)
        for base, B in (("X1", X1), ("W5-G", W5), ("K2", x["K2"].to_numpy(float))):
            lb = K.qlike(rv, B)
            for w in ("M", "h1", "h2", "M_bez_2022", "h2b22"):
                m = per[w]
                rows.append(dict(block="PH2", cand=name, window=w, group=base, metric="d_pct", value=float((l[m].mean() / lb[m].mean() - 1) * 100)))
            ci = EVX.delta_ci(l, lb, float(lb.mean()), b=ST.B, block=ST.BLOCK, seed=ST.SEED)
            dm = EVX.dm_one(rv, B, F)
            rows += [dict(block="PH2", cand=name, window="M", group=base, metric="lo90", value=float(ci["lo90"])),
                     dict(block="PH2", cand=name, window="M", group=base, metric="hi90", value=float(ci["hi90"])),
                     dict(block="PH2", cand=name, window="M", group=base, metric="p_dm_one", value=float(dm["p_one"]))]
        lx = K.qlike(rv, X1)
        for y in range(2018, 2027):
            m = per[str(y)]
            rows.append(dict(block="PH2", cand=name, window=str(y), group="X1", metric="d_pct", value=float((l[m].mean() / lx[m].mean() - 1) * 100)))
        # вклад лет в Δ к X1 окна M: доля лучшего года
        c = np.array([(l[per[str(y)]] - lx[per[str(y)]]).sum() for y in range(2018, 2027)]) / (len(x) * lx.mean()) * 100
        rows += [dict(block="PH2", cand=name, window="M", group="X1", metric="years_better", value=float((c < 0).sum())),
                 dict(block="PH2", cand=name, window="M", group="X1", metric="best_year_share_of_gain", value=float(c.min() / c[c < 0].sum()) if (c < 0).any() else float("nan")),
                 dict(block="PH2", cand=name, window="M", group="X1", metric="sum_worse_pp", value=float(c[c > 0].sum()))]
    return rows
