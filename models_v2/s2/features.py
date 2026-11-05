"""Признаки этапа 2 (Ж-197, Б2–Б3). Всё в %^2; строка t использует только данные дня t и раньше."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import daily, har

NAMES = ("ln_rv", "ln_rv_w", "ln_rv_m", "ln1p_j", "ln_bv", "ln_rv_h1", "ln_rv_h2", "rs_plus", "rs_minus",
         "gap_rel", "dow", "har_b")
N_BASE, I_HAR, CORE = 11, 11, (0, 1, 2, 3)
MONO = (0, 1, 2)
MIN_PAIRS_HAR = 252
HOUR_FLOOR = 1e-6
H1_TO, H2_FROM = "10:50", "17:50"


@dataclass
class Fp:
    dates: np.ndarray
    seg: np.ndarray
    rv: np.ndarray
    X: np.ndarray  # n x 12

    @property
    def lnrv(self) -> np.ndarray:
        return np.log(self.rv)


def _intraday(bars: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    main = daily.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = daily.intraday_returns(main)
    r2 = (100.0 * r["r"]) ** 2
    t = main["begin"].dt.strftime("%H:%M")
    day = r["date"]
    out = pd.DataFrame(index=idx)
    out["h1"] = r2[t <= H1_TO].groupby(day[t <= H1_TO]).sum()
    out["h2"] = r2[t >= H2_FROM].groupby(day[t >= H2_FROM]).sum()
    out["rs_plus"] = (r2 * (r["r"] > 0)).groupby(day).sum()
    out["rs_minus"] = (r2 * (r["r"] < 0)).groupby(day).sum()
    out["open1"] = main.groupby(main["begin"].dt.normalize())["open"].first()
    out["close_last"] = main.groupby(main["begin"].dt.normalize())["close"].last()
    return out


def har_feature(p: har.Panel, min_pairs: int = MIN_PAIRS_HAR) -> np.ndarray:
    """ln sigma2_b HAR-J на строке t по данным <= t (ежедневная OLS этапа 1); NaN при < min_pairs пар."""
    out = np.full(len(p.rv), np.nan)
    cols = list(har.COLS_HARJ)
    for i in range(len(p.rv)):
        if not np.isfinite(p.X[i]).all():
            continue
        Xt, yt = har.pairs(p, i, har.COLS_HARJ)
        if len(yt) < min_pairs:
            continue
        beta, s2, _ = har.ols(Xt, yt)
        out[i] = float(beta[0] + beta[1:] @ p.X[i, cols]) + s2 / 2
    return out


def build(bars: pd.DataFrame, min_pairs: int = MIN_PAIRS_HAR) -> Fp:
    dly, _ = daily.build_daily(bars)
    ix = _intraday(bars, dly.index).reindex(dly.index)
    p = har.make_panel(dly)
    X = np.full((len(dly), len(NAMES)), np.nan)
    X[:, :4] = p.X
    X[:, 4] = np.log(dly["bv"].to_numpy())
    X[:, 5] = np.log(np.maximum(ix["h1"].to_numpy(), HOUR_FLOOR))
    X[:, 6] = np.log(np.maximum(ix["h2"].to_numpy(), HOUR_FLOOR))
    X[:, 7], X[:, 8] = ix["rs_plus"].to_numpy(), ix["rs_minus"].to_numpy()
    prev = ix["close_last"].shift(1).to_numpy()
    same = np.concatenate([[False], p.seg[1:] == p.seg[:-1]])
    gap2 = (100.0 * np.log(ix["open1"].to_numpy() / prev)) ** 2
    X[:, 9] = np.where(same, gap2 / dly["rv_m"].to_numpy(), np.nan)
    X[:, 10] = dly.index.dayofweek.to_numpy()
    X[:, I_HAR] = har_feature(p, min_pairs)
    return Fp(p.dates, p.seg, p.rv, X)
