"""V4-A: сводка oracle по бумагам и группам (M16, SBER, ALL17) × окнам × варианту M (K2, K3, геометрическое среднее K2·K3). Только готовые прогнозы."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_io as IO, v4_oracle as OR

MS = {"K2": lambda a: a["k2"].to_numpy(float), "K3": lambda a: a["k3"].to_numpy(float), "KG": lambda a: np.sqrt(a["k2"].to_numpy(float) * a["k3"].to_numpy(float))}
KEYS = ["q_h", "q_m", "q_best", "o1_arith", "o1_geom", "o1", "o2", "o3", "o4"]


def oracle_table(tickers=None) -> pd.DataFrame:
    rows = []
    for t in tickers or IO.T17:
        a = IO.aligned(t)
        yr = a["t_plus"].dt.year
        for w, (lo, hi) in IO.WINDOWS.items():
            s = a[(yr >= lo) & (yr <= hi)]
            rv, h, k2, k3 = (s[c].to_numpy(float) for c in ("rv", "h", "k2", "k3"))
            o4 = OR.q(rv, OR.o4_series(rv, h, k2, k3))      # оболочка трёх {H, K2, K3} от выбора M не зависит
            for mname, fm in MS.items():
                rows.append({"ticker": t, "group": IO.P.group(t), "window": w, "m": mname, "n": len(s), **OR.oracle(rv, h, fm(s)), "o4": o4})
    return pd.DataFrame(rows)


def group_rows(t: pd.DataFrame) -> pd.DataFrame:
    """Группы: среднее по бумагам QLIKE-величин (части и доли пересчитаны из средних; веса — средние по бумагам)."""
    out = []
    for gname, sel in (("M16", t["group"] == "M16"), ("ALL17", t["group"].notna())):      # SBER — строка бумаги
        for (w, m), g in t[sel].groupby(["window", "m"], sort=False):
            o = {k: float(g[k].mean()) for k in KEYS + ["w_arith", "w_geom", "b_mult"]}
            out.append({"ticker": gname, "group": gname, "window": w, "m": m, "n": int(g["n"].sum()), **o, **OR.parts(o)})
    return pd.DataFrame(out)


def full_table(tickers=None) -> pd.DataFrame:
    t = oracle_table(tickers)
    return pd.concat([t, group_rows(t)], ignore_index=True)


def check_rows(t: pd.DataFrame) -> list:
    """Нарушения порядка: QLIKE >= 0; O2 <= O1 <= лучшей одиночной; O3 <= O2; O4 <= O2; части дают q_best."""
    bad, e = [], 1e-12
    for c in KEYS:
        if (t[c] < -e).any():
            bad.append("QLIKE < 0 в " + c)
    if (t["o2"] > t["o1"] + e).any() or (t["o1"] > t["q_best"] + e).any() or (t["o3"] > t["o2"] + e).any() or (t["o4"] > t["o2"] + e).any():
        bad.append("нарушен порядок O4 <= O2 <= O1 <= лучшая одиночная или O3 <= O2")
    s = t["p1_const_weight"] + t["p2_conditional"] + t["p3_out_of_range"] + t["p4_residual"]
    if not np.allclose(s, t["q_best"], rtol=0, atol=1e-10):
        bad.append("части не дают q_best")
    return bad
