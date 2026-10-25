"""Данные в памяти: дневные бары и σ_pred одного тикера (только чтение).

Повторяет загрузку V4 (`process_ticker_v4`, V4:1040-1103) для daily.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt.signals.interface import TickerData

MAIN = Path(os.environ.get("FASTBT_MAIN",
                           "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
STRAT = MAIN / "moex_volatility" / "results" / "final" / "strategies"
DATA_ROOT = Path(os.environ.get("FASTBT_DATA_ROOT", STRAT / "data"))   # --data-root
OHLCV = DATA_ROOT / "ohlcv_daily_full.parquet"
VPRED = Path(os.environ.get("FASTBT_VPRED", DATA_ROOT / "vpred_aligned.parquet"))
POS_PRE2026 = STRAT / "walkforward_v4" / "daily_positions_pre2026.parquet"

TICKERS = sorted(["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
                  "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"])


@lru_cache(maxsize=1)
def daily_df() -> pd.DataFrame:
    d = pd.read_parquet(OHLCV)
    d["date"] = pd.to_datetime(d["date"])
    return d


@lru_cache(maxsize=1)
def vpred_df() -> pd.DataFrame:
    v = pd.read_parquet(VPRED)
    v["date"] = pd.to_datetime(v["date"])
    return v


def align_sigma(dates: np.ndarray, ticker: str) -> dict:
    vp = vpred_df()
    vp = vp[vp["ticker"] == ticker].sort_values("date")
    vp_dates = pd.to_datetime(vp["date"].values)
    out = {}
    n = len(dates)
    for h in ("h1", "h5", "h22"):
        col = f"sigma_{h}"
        if col not in vp.columns:
            continue
        vals = vp[col].values
        arr = np.full(n, np.nan)
        idx = np.searchsorted(vp_dates, pd.to_datetime(dates), side="right") - 1
        ok = idx >= 0
        arr[ok] = vals[idx[ok]]
        out[h] = arr
    return out


CORP = Path(__file__).resolve().parent.parent / "data" / "corporate_actions.csv"


def corp_actions(ticker: str) -> pd.DataFrame:
    """Применяемые корпоративные события тикера (status == applied)."""
    ca = pd.read_csv(CORP, parse_dates=["date"])
    return ca[(ca["ticker"] == ticker) & (ca["status"] == "applied")]


@lru_cache(maxsize=64)
def load_ticker(ticker: str, corp: bool = False) -> TickerData:
    """corp=True — цены до даты события × factor, объём ÷ factor (data/corporate_actions.csv)."""
    d = daily_df()
    x = d[d["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    c, h, l, o, v = [x[k].values.astype(np.float64) for k in
                     ("close", "high", "low", "open", "volume")]
    if corp:
        for _, ev in corp_actions(ticker).iterrows():
            m = (x["date"] < ev["date"]).values
            f = float(ev["factor"])
            for a in (c, h, l, o):
                a[m] *= f
            v[m] /= f
    n = len(c)
    lr = np.zeros(n)
    lr[:-1] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    dates = pd.to_datetime(x["date"].values).values
    return TickerData(ticker, dates, o, h, l, c, v, lr, align_sigma(dates, ticker))


def year_bounds(dates: np.ndarray, year: int):
    """Ровно SW._year_bounds: (индекс 1 янв Y, индекс 1 янв Y+1)."""
    s = np.searchsorted(dates, np.datetime64(f"{year}-01-01"))
    e = np.searchsorted(dates, np.datetime64(f"{year + 1}-01-01"))
    return int(s), int(e)
