"""Ж-214, лист 1: синхронные M10-величины SBER–IMOEX: реализованная ковариация, семиковариации (Bollerslev–Li–Patton–Quaedvlieg),
корреляция, бета, семивариации рынка. Только бары основной сессии, присутствующие в ОБОИХ рядах с одним begin.

Доходность бара: close-к-close, если предыдущий слот (begin − 10 мин) того же дня тоже общий; иначе ln(close/open) этого бара
(возврат за ≤ 10 мин; ход через дыру в одном из рядов не входит ни в один ряд). Так обе доходности покрывают один интервал.
cov = cov_pp + cov_nn + cov_mix (pp: оба > 0, nn: оба < 0, mix: знаки разные; нулевая доходность вклада не даёт).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_sched as S

_SLOT = pd.Timedelta(minutes=S.BAR_MIN)


def sync_returns(sb: pd.DataFrame, ix: pd.DataFrame) -> pd.DataFrame:
    """Основные сессии SBER и IMOEX -> строки: date, begin, rs (SBER), rm (IMOEX) на общих барах."""
    a, b = (D.main_session(x)[["begin", "open", "close"]].sort_values("begin") for x in (sb, ix))
    c = a.merge(b, on="begin", suffixes=("_s", "_m")).sort_values("begin").reset_index(drop=True)
    day = c["begin"].dt.normalize()
    adj = (c["begin"].shift(1) == c["begin"] - _SLOT) & (day.shift(1) == day)
    out = pd.DataFrame({"date": day, "begin": c["begin"]})
    for k in ("s", "m"):
        lc = np.log(c["close_" + k].to_numpy(float))
        own = lc - np.log(c["open_" + k].to_numpy(float))
        out["r" + k] = np.where(adj, lc - np.r_[np.nan, lc[:-1]], own)
    return out


def aggregate(sr: pd.DataFrame) -> pd.DataFrame:
    s, m = sr["rs"].to_numpy(float), sr["rm"].to_numpy(float)
    pp, nn = (s > 0) & (m > 0), (s < 0) & (m < 0)
    mix = (s * m) < 0
    f = pd.DataFrame({"date": sr["date"], "n_sync": 1, "rv_s": s * s, "rv_m": m * m, "cov": s * m, "cov_pp": np.where(pp, s * m, 0.0),
                      "cov_nn": np.where(nn, s * m, 0.0), "cov_mix": np.where(mix, s * m, 0.0), "rsm_pos": np.where(m > 0, m * m, 0.0),
                      "rsm_neg": np.where(m < 0, m * m, 0.0)})
    g = f.groupby("date").sum()
    cols = [c for c in g.columns if c != "n_sync"]
    g[cols] *= D.SCALE
    g["corr"] = g["cov"] / np.sqrt(g["rv_s"] * g["rv_m"])
    g["beta"] = g["cov"] / g["rv_m"]
    return g.rename(columns={"rv_s": "rv_s_sync", "rv_m": "rv_m_sync"})


def build(sb: pd.DataFrame, ix: pd.DataFrame, index) -> pd.DataFrame:
    """На канонических датах `index`; день без общих баров — NaN."""
    return aggregate(sync_returns(sb, ix)).reindex(index)


def identity(t: pd.DataFrame) -> float:
    """max |cov_pp + cov_nn + cov_mix − cov| / sqrt(rv_s·rv_m)."""
    num = (t["cov_pp"] + t["cov_nn"] + t["cov_mix"] - t["cov"]).abs()
    return float((num / np.sqrt(t["rv_s_sync"] * t["rv_m_sync"])).max())
