"""V4-B: панель общих дат H ∩ K2 ∩ K3 (+ D2) по 17 бумагам и широкая таблица оценки для `hp_stats_sel.crit_panel` (колонки: ticker, date, t_plus, group, rv_tplus, ряды σ̂²)."""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_io as IO

CTRL = {"k1": "h", "k2": "k2", "k3": "k3", "d2": "d2"}


@functools.lru_cache(maxsize=4)
def _panel(tickers: tuple) -> pd.DataFrame:
    fr = []
    for t in tickers:
        a = IO.aligned(t)
        a.insert(0, "ticker", t)
        fr.append(a)
    return pd.concat(fr, ignore_index=True)


def panel(tickers=None) -> pd.DataFrame:
    """Одна таблица: ticker, date, t_plus, rv, h, k2, k3, d2 (NaN, где D2 нет). Копия — вызывающий вправе менять."""
    return _panel(tuple(tickers or IO.T17)).copy()


def eval_frame(P: pd.DataFrame, series: dict | None = None) -> pd.DataFrame:
    """Широкая таблица для crit_panel/статистики. series — {имя: таблица прогнозов с ticker, date, F}; контроли k1, k2, k3, d2 добавляются из P."""
    k = pd.DataFrame({"ticker": P["ticker"].to_numpy(), "date": P["date"].to_numpy(), "t_plus": P["t_plus"].to_numpy(), "group": [IO.P.group(t) for t in P["ticker"]],
                      "rv_tplus": P["rv"].to_numpy(float)})
    for name, col in CTRL.items():
        k[name] = P[col].to_numpy(float)
    for name, f in (series or {}).items():
        k = k.merge(f[["ticker", "date", "F"]].rename(columns={"F": name}), on=["ticker", "date"], how="left")
    return k.sort_values(["ticker", "date"]).reset_index(drop=True)


def crit_table(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    """`hp_stats_sel.crit_panel` — как есть (определение Ж-217); без пересчёта своим средним."""
    return S.crit_panel(k, cols)
