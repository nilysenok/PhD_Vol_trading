"""V6-0, G6: ряд остатка HAR-J e_t = ln(RV_{t⁺} / H_t), H = K1 (HAR-J(б) PIT SBER). Только описательная статистика по годам (среднее, СКО ddof = 1, автокорреляция 1 и 5); признаки V6 на остаток не регрессируются.
Прогрев по коду K1: `models_v2/har.py` — расширяющееся окно OLS по парам t+1 ≤ i (`pairs`), один сегмент; прогноз с i воспроизводим `forecast_at` для любой даты с набранным окном (MIN_PAIRS). Сохранённый H начинается с d = 2016-12-30;
для 2014–2016 восстановлена только возможность (число дат, сверка на 2016-12-30 и 2017-01-03 побитно), ряд e до 2017 лист 0 не строит (замороженный H не пересчитывается)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s10 import hp_har, hp_paths as P

LAGS = (1, 5)


def acf(x: np.ndarray, k: int) -> float:
    x = np.asarray(x, float)
    if len(x) <= k + 2:
        return float("nan")
    x = x - x.mean()
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


def stat_row(e: np.ndarray, label, extra=None) -> dict:
    return dict(section="stat", window=label, n=len(e), mean=float(e.mean()), std=float(e.std(ddof=1)), acf1=acf(e, 1), acf5=acf(e, 5), **(extra or {}))


def resid_series(w: pd.DataFrame) -> pd.Series:
    return pd.Series(np.log(w["rv"].to_numpy(float) / w["K1"].to_numpy(float)), index=w.index, name="e")


def stats_rows(w: pd.DataFrame) -> pd.DataFrame:
    e = resid_series(w).to_numpy()
    y = pd.DatetimeIndex(w["t_plus"]).year.to_numpy()
    rows = [stat_row(e[y == k], str(k)) for k in range(2017, 2027)]
    rows += [stat_row(e[(y >= 2017) & (y <= 2021)], "2017_2021"), stat_row(e[(y >= 2022) & (y <= 2025)], "2022_2025"), stat_row(e, "all")]
    return pd.DataFrame(rows)


def warmup_rows(w: pd.DataFrame) -> pd.DataFrame:
    """Сколько закрытых пар доступно на 01.01.2017, сколько дат прогрева 2014–2016 имеют PIT-прогноз H; сверка воспроизведения `forecast_at` с сохранённым K1 на первых двух датах."""
    daily = pd.read_parquet(P.OUT / "daily_SBER.parquet")
    daily.index = pd.to_datetime(daily.index)
    p = H.make_panel(daily)
    har = pd.read_parquet(P.OUT / "har_SBER.parquet")
    har["d"] = pd.to_datetime(har["d"])
    first = har.iloc[0]
    i0 = int(np.where(p.dates == np.datetime64(first["d"]))[0][0])
    n_pairs = len(H.pairs(p, i0, H.COLS_HARJ)[1])
    chk = [abs(H.forecast_at(p, int(np.where(p.dates == np.datetime64(har.iloc[j]["d"]))[0][0]))["harj_b"] - har.iloc[j]["sigma2"]) for j in (0, 1)]
    grid = hp_har.grid(p, "2014-01-01", P.MIN_PAIRS)
    warm = grid[p.dates[grid] < np.datetime64("2016-12-30")]
    rows = [dict(section="warmup", item="первая дата d сохранённого H (K1)", value=str(first["d"].date())), dict(section="warmup", item="закрытых пар в окне K1 на d = первая дата (n_pairs файла)", value=int(first["n_pairs"])),
            dict(section="warmup", item="то же, пересчёт har.pairs на этой дате", value=n_pairs), dict(section="warmup", item="max|forecast_at − сохранённый sigma2| на двух первых датах", value=float(max(chk))),
            dict(section="warmup", item="MIN_PAIRS (минимум пар для прогноза)", value=int(P.MIN_PAIRS)), dict(section="warmup", item="дат прогрева до 2016-12-30 с PIT-прогнозом H (hp_har.grid, пар ≥ MIN_PAIRS)", value=int(len(warm))),
            dict(section="warmup", item="первая такая дата", value=str(pd.Timestamp(p.dates[warm[0]]).date()) if len(warm) else "—"),
            dict(section="warmup", item="закрытых пар, доступных к 01.01.2017 (последняя дата d ≤ 31.12.2016)", value=int(har[har["d"] <= "2016-12-31"].iloc[-1]["n_pairs"]))]
    return pd.DataFrame(rows)


def resid_table(w: pd.DataFrame) -> tuple:
    wu, st = warmup_rows(w), stats_rows(w)
    return pd.concat([wu, st], ignore_index=True), dict(warm=wu, stats=st, e=resid_series(w))
