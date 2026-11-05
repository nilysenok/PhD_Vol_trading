"""Численная устойчивость кастомного QLIKE (GATES_B_xgb P5): g/h против конечных разностей, сравнение с reg:gamma на 12 датах, признак нестабильной строки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s4 import rows
from models_v2.s5.b1 import jobs, losses
from models_v2.s6.xgb import cfgx, worldx

TOL_FD = 1e-6
RATIO_LO, RATIO_HI = 0.1, 10.0


def fd_error(y, F, h: float = 1e-4) -> float:
    """Макс. |аналитика − конечные разности| / (1 + |разности|) по g и h."""
    g, hs = losses.grad_hess(y, F)
    gn, hn = losses.numeric_grad_hess(y, F, h)
    return float(max(np.max(np.abs(g - gn) / (1 + np.abs(gn))), np.max(np.abs(hs - hn) / (1 + np.abs(hn)))))


def fd_grid() -> tuple:
    yy, FF = np.meshgrid(np.logspace(-3, 3, 25), np.linspace(-6, 6, 25))
    return yy.ravel(), FF.ravel()


def stab_dates(F, grid) -> list:
    """Первая дата сетки каждого года и 22.02.2022 (последняя d перед целью 24.02.2022)."""
    yr = F.dates[grid].astype("datetime64[Y]").astype(int) + 1970
    out = [int(grid[np.flatnonzero(yr == y)[0]]) for y in sorted(set(yr))]
    feb = np.flatnonzero(F.dates == np.datetime64("2022-02-22"))
    if len(feb) and int(feb[0]) in set(grid.tolist()):
        out.append(int(feb[0]))
    return sorted(set(out))


def compare_dates(w: dict, ds=None) -> pd.DataFrame:
    """На каждой d: custom QLIKE и reg:gamma на одном окне и HP года. Строка: d, loss, k, sigma2, finite, tr_loss, es_loss."""
    F = w["F"]
    ds = stab_dates(F, w["grid"]) if ds is None else ds
    sp = {l: worldx.spec_of(w, cfgx.mk("s_" + l, l)) for l in ("qlike", "gamma")}
    out = []
    for d in ds:
        hp = w["i2"][rows.year_of(F, int(d))]
        for l, s in sp.items():
            m = jobs.fit_at(F, s, jobs.pairs(F, s, int(d)), hp)
            f = float(m["predict"](s.X[[d]])[0])
            s2 = float(np.exp(s.off[d] + f))
            out.append({"d": F.dates[d], "loss": l, "k": m["k"], "sigma2": s2, "finite": bool(np.isfinite(s2) and s2 > 0),
                        "tr_loss": m["tr_loss"], "es_loss": m["es_loss"]})
    return pd.DataFrame(out)


def pair_diff(df: pd.DataFrame) -> pd.DataFrame:
    """По датам: |Δ ln σ̂²| между qlike и gamma и разность k."""
    a, b = (df[df["loss"] == l].set_index("d") for l in ("qlike", "gamma"))
    return pd.DataFrame({"abs_dln": (np.log(a["sigma2"]) - np.log(b["sigma2"])).abs(), "dk": a["k"] - b["k"]})


def unstable(fc: pd.DataFrame, ref: np.ndarray) -> bool:
    """P5(в): σ̂² не конечен, <= 0 или отношение к σ̂² опорной строки вне [0,1; 10]."""
    s2 = fc["sigma2"].to_numpy(float)
    bad = ~np.isfinite(s2) | (s2 <= 0)
    r = s2 / np.asarray(ref, float)
    return bool(bad.any() or (r < RATIO_LO).any() or (r > RATIO_HI).any())
