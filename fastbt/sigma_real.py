"""Реализованная σ этапа 1 (Ж-047 п. 2): √RV за 5/22/66 дней и EWMA λ = 0,94, только данные ≤ d до 18:50.

Прогнозных рядов здесь нет (тест `test_sigma_real.py::test_static_no_forecast_imports` разбирает импорты).
 - RV дня d = сумма квадратов 10-минутных лог-доходностей внутри дня по баров с началом в [09:50; 18:50)
   (границы — константы `daily_1850`, Н6); доходность через ночь не считается;
 - rv5/rv22/rv66: σ_d = √(среднее RV за w дней, оканчивающихся днём d) — день d входит, он известен на 18:50;
   решение на d исполняется на d+1 (Н6), поэтому заглядывания вперёд нет;
 - ewma094: v_d = λ·v_{d−1} + (1−λ)·RV_d, v_первый = RV_первый, σ_d = √v_d; ряд даётся с EWMA_MIN-го наблюдения;
 - нет истории — NaN, не подстановка (Д4).
❓6 (RV в features_pit_v3 до 18:50?) не подтверждён (отчёт К1): RV строится из 10-минуток здесь, `features_pit_v3.rv` не берётся.
Данные ≥ 2022 не читаются (фильтр на чтении, `daily_1850.END`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import zona
from fastbt.daily_1850 import CUT_MIN, END, OPEN_MIN, default_main, read_bars  # noqa: F401  (END — реэкспорт)

WINDOWS = {"rv5": 5, "rv22": 22, "rv66": 66}
LAMBDA = 0.94
EWMA_MIN = 22
NAMES = (*WINDOWS, "ewma094")


def rv_1850(bars: pd.DataFrame, cut_min: int = CUT_MIN) -> pd.Series:
    """RV по дням (индекс date) из 10-минуток (timestamp, close): бары с началом в [OPEN_MIN; cut_min)."""
    b = bars.sort_values("timestamp")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= OPEN_MIN) & (mod < cut_min)]
    day = b["timestamp"].dt.normalize()
    r2 = np.log(b["close"].astype("float64")).groupby(day).diff() ** 2
    rv = r2.groupby(day).sum(min_count=1).dropna()
    rv.index.name = "date"
    return rv


def sigma_rv(rv: pd.Series, w: int) -> pd.Series:
    """√(среднее RV за w дней, оканчивающихся днём d); первые w−1 значений — NaN."""
    return np.sqrt(rv.rolling(w, min_periods=w).mean())


def sigma_ewma(rv: pd.Series, lam: float = LAMBDA) -> pd.Series:
    """√v_d, v_d = λ·v_{d−1} + (1−λ)·RV_d; до EWMA_MIN-го наблюдения — NaN."""
    v = rv.ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(v.where(rv.notna().cumsum() >= EWMA_MIN))


def sigma_all(rv: pd.Series) -> dict[str, pd.Series]:
    return {**{n: sigma_rv(rv, w) for n, w in WINDOWS.items()}, "ewma094": sigma_ewma(rv)}


def load_rv(tickers: list[str], main: Path | None = None) -> pd.DataFrame:
    """RV по тикерам из 10-минуток основной копии (только чтение, < 2022): колонки date, ticker, rv."""
    src = (main or default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    parts = []
    for t in tickers:
        s = rv_1850(read_bars(src / f"{t}.parquet"))          # граница — зона (zona.end())
        parts.append(s.rename("rv").reset_index().assign(ticker=t))
    d = pd.concat(parts, ignore_index=True)
    return d[d["date"] < zona.end()][["date", "ticker", "rv"]].sort_values(["ticker", "date"]).reset_index(drop=True)


def sigma_frame(rv: pd.DataFrame, name: str) -> pd.DataFrame:
    """(date, ticker, sigma) для ряда `name` из long-таблицы RV (date, ticker, rv)."""
    assert name in NAMES, name
    parts = []
    for t, g in rv.sort_values("date").groupby("ticker"):
        r = g.set_index("date")["rv"]
        s = sigma_ewma(r) if name == "ewma094" else sigma_rv(r, WINDOWS[name])
        parts.append(s.rename("sigma").reset_index().assign(ticker=t))
    d = pd.concat(parts, ignore_index=True).dropna()
    return d[["date", "ticker", "sigma"]].sort_values(["date", "ticker"]).reset_index(drop=True)


def proverka_sdvig(sigma: pd.Series, rv: pd.Series, ks: tuple = (-2, -1, 0, 1)) -> dict:
    """Д1 для окон по приращениям: corr(Δσ²_d, RV_{d+k}); пик обязан быть на k = 0 (последний известный день).
    Сдвиг ряда на +1 день даёт пик при k = +1 — ok = False."""
    dv = (sigma ** 2).diff()
    c = {k: float(dv.corr(rv.shift(-k))) for k in ks}
    best = max(c, key=lambda k: c[k])
    return {"argmax": best, "ok": best == 0, **{f"corr_k{k:+d}": v for k, v in c.items()}}
