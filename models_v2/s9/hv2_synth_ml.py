"""Ж-214, лист 4: синтетический DWorld (M10-величины, синхронные с IMOEX, ряд IMOEX) с известной структурой для тестов V9/V10. Не читает данные.
signal > 0: ln RV_{t+1} зависит от ln(RS⁻/RS⁺)_t (группа S) — HAR этого не видит, структурная коррекция обязана улучшить QLIKE."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC, hv2_msfeat as MS, hv2_world as W


def dworld(n: int = 520, tail: int = 60, seed: int = 5, signal: float = 0.0) -> DC.DWorld:
    rng = np.random.default_rng(seed)
    sgn = rng.normal(0, 0.6, n)
    ln = np.zeros(n)
    for t in range(1, n):
        ln[t] = 0.85 * ln[t - 1] + signal * sgn[t - 1] + rng.normal(0, 0.3)
    rv = np.exp(ln + 1.0)
    u, sh = 1 / (1 + np.exp(sgn)), rng.dirichlet([4, 8, 4], n)
    j = rv * rng.uniform(0, 0.35, n) * (rng.random(n) < 0.6)
    rm, rs, corr = rv * rng.uniform(0.3, 0.8, n), rv * rng.uniform(0.9, 1.0, n), rng.uniform(0.2, 0.9, n)
    cov = corr * np.sqrt(rs * rm)
    mix = -cov * rng.uniform(0, 0.2, n)
    pp = (cov - mix) * rng.uniform(0.3, 0.7, n)
    sp = rng.uniform(0.3, 0.7, n)
    m10 = pd.DataFrame({"rv": rv, "j": j, "c": rv - j, "rs_pos": rv * u, "rs_neg": rv * (1 - u), "rv_open": rv * sh[:, 0], "rv_mid": rv * sh[:, 1], "rv_close": rv * sh[:, 2],
                        "corr": corr, "beta": cov / rm, "cov": cov, "cov_pp": pp, "cov_nn": cov - mix - pp, "cov_mix": mix, "rv_s_sync": rs, "rv_m_sync": rm,
                        "rsm_pos": rm * sp, "rsm_neg": rm * (1 - sp)}, index=pd.bdate_range("2018-01-01", periods=n))
    seg = np.zeros(n, int)
    X = np.column_stack([np.log(rv), np.log(DC.wmean(rv, seg, 5)), np.log(DC.wmean(rv, seg, 22)), np.log1p(j)])
    p = H.Panel(m10.index.to_numpy("datetime64[ns]"), seg, rv, X)
    rq = rv ** 2 * np.exp(rng.normal(0, 0.3, n))
    dw = DC.build(W.World(p, rq, np.sqrt(2 * rq / 50) / rv, H.eligible(p)[-tail:]), m10, MS.KEYS)
    ir = pd.Series(rm * rng.uniform(0.9, 1.1, n))
    return dataclasses.replace(dw, comp={**dw.comp, "imo_rv": ir.to_numpy(), "imo_w": ir.rolling(5, min_periods=1).mean().to_numpy(), "imo_m": ir.rolling(22, min_periods=1).mean().to_numpy()})
