"""Сводка по позициям: МСП по годам, ΔШарп C−A / D−A, парный бутстрап (FB:25-35)."""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from fastbt import msp


def rate_for(cfg) -> pd.Series:
    return msp.rate_cbr() if cfg.fix_rate else msp.rate_restored()


def portfolios(pos: pd.DataFrame, cfg, years) -> dict:
    pos = pos[(pos["test_year"] >= years[0]) & (pos["test_year"] <= years[-1])]
    rate = rate_for(cfg)
    return {ap: msp.portfolio(pos[pos["approach"] == ap], rate, cfg.msp_comm)
            for ap in sorted(pos["approach"].unique())}


def _sh(x, axis=-1):
    s = np.std(x, axis=axis, ddof=1)
    return np.where(s > 1e-12, np.mean(x, axis=axis) * sqrt(252) / np.where(s > 0, s, 1), 0.0)


def boot_delta(a, b, reps, seed=42):
    """ΔШарп(a) − Шарп(b) на общих днях; p = доля Δ* ≤ 0; ДИ 95%."""
    n = len(a)
    rng = np.random.RandomState(seed)
    d = np.empty(reps)
    for i0 in range(0, reps, 500):
        k = min(500, reps - i0)
        idx = rng.randint(0, n, (k, n))
        d[i0:i0 + k] = _sh(a[idx]) - _sh(b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(_sh(a) - _sh(b)), float(np.mean(d <= 0)), float(lo), float(hi)


def summary(pos: pd.DataFrame, cfg, years, reps=1000):
    """(таблица Шарпов по годам и подходам, таблица Δ с бутстрапом)."""
    ports = portfolios(pos, cfg, years)
    rows = []
    for ap, p in ports.items():
        r = {"approach": ap, "all": msp.sharpe(p["excess"])}
        for y in years:
            ex = p.loc[p.index.year == y, "excess"]
            r[str(y)] = msp.sharpe(ex) if len(ex) > 2 else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows)
    deltas = []
    for x in ("B", "C", "D"):
        if x in ports and "A" in ports:
            j = ports[x].index.intersection(ports["A"].index)
            a, b = ports[x].loc[j, "excess"].values, ports["A"].loc[j, "excess"].values
            obs, p, lo, hi = boot_delta(a, b, reps)
            r = {"delta": f"{x}-A", "obs": obs, "p": p, "ci_lo": lo, "ci_hi": hi, "days": len(j)}
            for y in years:
                m = j.year == y
                if m.sum() > 2:
                    r[str(y)] = msp.sharpe(a[m]) - msp.sharpe(b[m])
            deltas.append(r)
    return tab, pd.DataFrame(deltas)
