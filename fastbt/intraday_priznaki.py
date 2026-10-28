"""Ж-097: внутридневные признаки по 10-минуткам (только данные [09:50; 18:50) дня d) и таблица информации без стратегий.

f1 = RSV⁻/RV; f2 = (close − low)/(high − low); f3 = max(0, 1 − BV/RV), BV = (π/2)·Σ|r_i||r_{i−1}|;
f4 = объём баров с началом ≥ 17:50 / объём дня; f5 = sign(close_d/close_{d−1} − 1)·f2.
Терцили — по тикеру, скользящее окно WIN по данным до d−1 (shift(1)). Данные ≥ 2020 не читаются (END).
Гипотез здесь нет: это карта информации (лёгкий контур), не отбор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.daily_1850 import CUT_MIN, OPEN_MIN, read_bars

END = pd.Timestamp("2020-01-01")          # train4 2016–2019 + прогрев 2015; 2020+ не читается
FEATURES = ("f1", "f2", "f3", "f4", "f5")
WIN, MIN_OBS, LAST_HOUR = 252, 240, 60


def features_ticker(bars: pd.DataFrame, cut_min: int = CUT_MIN) -> pd.DataFrame:
    """Признаки по дням (индекс date) + open10 (open бара 10:00), close (последний бар < cut_min)."""
    b = bars.sort_values("timestamp")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    keep = (mod >= OPEN_MIN) & (mod < cut_min)
    b, mod = b[keep].copy(), mod[keep]
    day = b["timestamp"].dt.normalize()
    r = np.log(b["close"].astype("float64")).groupby(day).diff()
    b["r2"], b["r2n"] = r ** 2, (r ** 2).where(r < 0, 0.0).where(r.notna())
    b["bvt"] = r.abs() * r.abs().groupby(day).shift(1)
    b["vlast"] = b["volume"].astype("float64").where(mod >= cut_min - LAST_HOUR, 0.0)
    g = b.groupby(day)
    rv, hi, lo, cl = g["r2"].sum(min_count=1), g["high"].max(), g["low"].min(), g["close"].last()
    out = pd.DataFrame(index=rv.index)
    out["f1"] = g["r2n"].sum(min_count=1) / rv
    out["f2"] = ((cl - lo) / (hi - lo)).where(hi > lo)
    out["f3"] = (1.0 - np.pi / 2 * g["bvt"].sum(min_count=1) / rv).clip(lower=0.0)
    out["f4"] = g["vlast"].sum() / g["volume"].sum().astype("float64")
    out["f5"] = np.sign(cl / cl.shift(1) - 1.0) * out["f2"]
    o10 = b[mod == 10 * 60].groupby(day[mod == 10 * 60])["open"].first()
    out["open10"], out["close"] = o10.reindex(out.index), cl
    out.index.name = "date"
    return out.replace([np.inf, -np.inf], np.nan)


def load_features(tickers: list[str], src, start: str = "2015-01-01") -> dict:
    """{ticker: features_ticker} из candles_10m (src — папка с <TICKER>.parquet), бары < END."""
    res = {}
    for t in tickers:
        d = features_ticker(read_bars(src / f"{t}.parquet", END))
        res[t] = d[d.index >= pd.Timestamp(start)]
    return res


def pit_f1_f3(path) -> pd.DataFrame:
    """Ж-100: f1 = rsv_neg/rv и f3 = jump/rv (jump = max(0, rv − bv), т.е. 1 − BV/RV с нижней границей 0) из готового
    features_pit_v3 (date < END). Индекс (ticker, date)."""
    p = pd.read_parquet(path, columns=["date", "ticker", "rv", "rsv_neg", "jump"], filters=[("date", "<", END.to_pydatetime())])
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] < END].set_index(["ticker", "date"])
    return pd.DataFrame({"f1": p["rsv_neg"] / p["rv"], "f3": p["jump"] / p["rv"]}).replace([np.inf, -np.inf], np.nan)


def with_pit(feats: dict, pit: pd.DataFrame) -> dict:
    """Заменяет f1, f3 в {ticker: features} готовыми из pit (пересчёта из баров нет); f2, f4, f5 остаются по барам."""
    out = {}
    for t, d in feats.items():
        d = d.copy()
        p = pit.loc[t].reindex(d.index)
        d["f1"], d["f3"] = p["f1"], p["f3"]
        out[t] = d
    return out


def tercile_cell(x: pd.Series) -> pd.Series:
    """0/1/2 = нижний/средний/верхний терциль x_d по границам окна WIN из x_{d−WIN..d−1} (shift(1)); нет окна — NaN."""
    lo = x.rolling(WIN, min_periods=MIN_OBS).quantile(1 / 3).shift(1)
    hi = x.rolling(WIN, min_periods=MIN_OBS).quantile(2 / 3).shift(1)
    c = pd.Series(np.where(x > hi, 2.0, np.where(x > lo, 1.0, 0.0)), index=x.index)
    return c.where(x.notna() & lo.notna() & hi.notna())


def forward_excess(d: pd.DataFrame, rf: pd.Series, k: int) -> pd.Series:
    """Доходность от open 10:00 дня d+1 до close дня d+k минус ставка·дни (rf — годовая доля по date, дни = кал. дни d+1…d+k)."""
    ret = d["close"].shift(-k) / d["open10"].shift(-1) - 1.0
    idx = d.index.to_series()
    cal = (idx.shift(-k) - idx.shift(-1)).dt.days + 1
    return ret - rf.reindex(d.index).shift(-1) / 365.0 * cal


def cell_stats(y: np.ndarray, cell: np.ndarray, day: np.ndarray) -> dict:
    """Средние по терцилям, n, спред верх−низ и t-статистики с кластеризацией по дню (G/(G−1)). cell ∈ {0,1,2}."""
    n = np.array([(cell == c).sum() for c in range(3)], float)
    if (n == 0).any():
        return dict(n=n, m=np.full(3, np.nan), spread=np.nan, t=np.nan, t_cell=np.full(3, np.nan))
    m = np.array([y[cell == c].mean() for c in range(3)])
    _, gi = np.unique(day, return_inverse=True)
    s = np.zeros((gi.max() + 1, 3))
    np.add.at(s, (gi, cell.astype(int)), y - m[cell.astype(int)])
    v = (s.T @ s) / np.outer(n, n) * (len(s) / max(len(s) - 1, 1))
    var_sp = v[2, 2] + v[0, 0] - 2 * v[0, 2]
    return dict(n=n, m=m, spread=m[2] - m[0], t=(m[2] - m[0]) / np.sqrt(var_sp) if var_sp > 0 else np.nan,
                t_cell=m / np.sqrt(np.diag(v)))


def panel(feats: dict, rfs: pd.Series, f: str, ks=(1, 5), t0="2016-01-01", t1="2019-12-31", shuffle=None) -> dict:
    """{k: (y, cell, day)} по всем тикерам для признака f; shuffle — np.random.Generator: признак перемешан по времени внутри тикера."""
    out = {k: ([], [], []) for k in ks}
    for t, d in feats.items():
        x = d[f].copy()
        if shuffle is not None:
            ok = x.notna().to_numpy()
            v = x.to_numpy().copy()
            v[ok] = shuffle.permutation(v[ok])
            x = pd.Series(v, index=x.index)
        cell = tercile_cell(x)
        for k in ks:
            y = forward_excess(d, rfs, k)
            m = (cell.notna() & y.notna() & (d.index >= t0) & (d.index <= t1)).to_numpy()
            out[k][0].append(y.to_numpy()[m]); out[k][1].append(cell.to_numpy()[m]); out[k][2].append(d.index.values[m])
    return {k: tuple(np.concatenate(a) for a in v) for k, v in out.items()}
