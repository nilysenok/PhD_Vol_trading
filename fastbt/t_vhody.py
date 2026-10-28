"""Сборка `inp` для детекторов T16–T20 из данных этапа 1 (К-16…К-20; контракт #376 §2.8). Единственное место чтения файлов.

Источники (все < 2022, обрезка по CUT на чтении):
- дневные бары Н6 (close/high/low/volume): `data/fastbt/daily/ohlcv_daily_1850.parquet`, `date ≤ CUT` фильтром pyarrow;
- σ̂ (rv5/rv22/rv66, известны на 18:50 d): 10-минутки `candles_10m/{T}.parquet`, `timestamp < CUT + 1 день` на чтении;
- IMOEX close 18:50 и sig_mkt: `rezhim_e1.imoex_daily` (фильтр < 2022 на чтении) + обрезка `date ≤ CUT`;
- дивиденды: csv из `fastbt.dividendy.TABLE`, строки `last_cum_date ≤ CUT`.
Ряды выровнены на даты тикера (с START = 2017-01-01); нет значения — NaN, без подстановки (Д4).
inp = {"dates", "tickers": {t: {"dates", "close", "high", "low", "vol", "sigma", "sigma5", "sigma66", "mkt_close", "mkt_sigma"}}}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fastbt import daily_1850 as d18, rezhim_e1, sigma_real

CUT = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2017-01-01")


def _ohlcv(path: Path) -> pd.DataFrame:
    t = pq.read_table(path, filters=[("date", "<=", CUT.to_pydatetime())]).to_pandas()
    t["date"] = pd.to_datetime(t["date"])
    if len(t) and t["date"].max() > CUT:
        raise PermissionError("T-входы: даты ≥ 2022")
    return t


def _sigmas(tickers, main: Path) -> dict:
    src = main / "moex_discovery" / "data" / "raw" / "candles_10m"
    out = {}
    for t in tickers:
        bars = d18.read_bars(src / f"{t}.parquet", CUT + pd.Timedelta(days=1))
        rv = sigma_real.rv_1850(bars)
        rv = rv[rv.index <= CUT]
        out[t] = sigma_real.sigma_all(rv)
    return out


def _imoex(path: Path | None) -> pd.Series:
    d = rezhim_e1.imoex_daily(path)
    d = d[pd.to_datetime(d["date"]) <= CUT]
    return d.set_index(pd.to_datetime(d["date"]))


def _al(s: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    return s.reindex(idx).to_numpy(float)


def zagruzit(tickers=None, main: Path | None = None, ohlcv: Path | None = None, imoex: Path | None = None) -> dict:
    """Собрать `inp` для всех тикеров (по умолчанию `fastbt.data.TICKERS`)."""
    if tickers is None:
        from fastbt.data import TICKERS
        tickers = TICKERS
    main = main or d18.default_main()
    bars = _ohlcv(ohlcv or d18.out_path(main))
    sg, im = _sigmas(tickers, main), _imoex(imoex)
    out = {}
    for t in tickers:
        b = bars[(bars["ticker"] == t) & (bars["date"] >= START)].sort_values("date")
        idx = pd.DatetimeIndex(b["date"])
        out[t] = dict(dates=idx.to_numpy("datetime64[D]"), close=b["close"].to_numpy(float), high=b["high"].to_numpy(float),
                      low=b["low"].to_numpy(float), vol=b["volume"].to_numpy(float),
                      sigma=_al(sg[t]["rv22"], idx), sigma5=_al(sg[t]["rv5"], idx), sigma66=_al(sg[t]["rv66"], idx),
                      mkt_close=_al(im["close"], idx), mkt_sigma=_al(im["sig_mkt"], idx))
    alld = np.unique(np.concatenate([x["dates"] for x in out.values()])) if out else np.array([], "datetime64[D]")
    return {"dates": alld, "tickers": out}


def dividendy_tablitsa(path: Path) -> pd.DataFrame:
    """Таблица дивидендов (`ticker, last_cum_date, announce_date, dps_rub`), строки с отсечкой ≤ CUT."""
    t = pd.read_csv(path, parse_dates=["last_cum_date", "announce_date"])
    t = t[t["last_cum_date"] <= CUT]
    return t[["ticker", "last_cum_date", "announce_date", "dps_rub"]].reset_index(drop=True)
