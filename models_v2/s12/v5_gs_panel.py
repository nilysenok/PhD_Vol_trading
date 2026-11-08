"""V5 лист 1a: панель на оси дат цели τ (t⁺) — матрицы RV, основы K8 (или H), маска «строка есть», месяц дня прогноза d. Только чтение замороженных входов листа 0; E11 во вход не попадает (17 бумаг `IO.T17`)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2.s11 import v4_rqc_calib as RQ
from models_v2.s12 import v5_0_io as IO


@dataclass
class Panel:
    tick: list
    axis: pd.DatetimeIndex          # даты цели τ
    rv: np.ndarray                  # (N, m) RV дня τ (NaN, где строки нет)
    h: np.ndarray                   # K1 = HAR-J
    m: np.ndarray                   # M = sqrt(K2·K3)
    k8: np.ndarray                  # K8 = sqrt(H·sqrt(K2·K3)) — формула V4
    pres: np.ndarray                # (N, m) bool: строка есть и RV, K1, K2, K3 конечны и > 0
    dd: np.ndarray                  # (N, m) datetime64: день прогноза d (NaT, где строки нет)
    mon: np.ndarray                 # (N, m) int: год·12 + месяц дня d (−1, где строки нет)


def build(k: pd.DataFrame | None = None, tickers: list | None = None) -> Panel:
    """Панель из общего кадра листа 0 (`IO.frame()`): колонки ticker, date (=d), t_plus (=τ), rv_tplus, k1, k2, k3."""
    k = IO.frame() if k is None else k
    tick = list(tickers or IO.T17)
    k = k[k["ticker"].isin(tick)]
    axis = pd.DatetimeIndex(sorted(k["t_plus"].unique()))
    pos = {d: i for i, d in enumerate(axis)}
    N, m = len(axis), len(tick)
    mat = {c: np.full((N, m), np.nan) for c in ("rv_tplus", "k1", "k2", "k3")}
    dd = np.full((N, m), np.datetime64("NaT"), dtype="datetime64[ns]")
    for j, t in enumerate(tick):
        g = k[k["ticker"] == t]
        r = g["t_plus"].map(pos).to_numpy()
        for c in mat:
            mat[c][r, j] = g[c].to_numpy(float)
        dd[r, j] = g["date"].to_numpy("datetime64[ns]")
    ok = np.ones((N, m), bool)
    for c in mat:
        ok &= np.isfinite(mat[c]) & (mat[c] > 0)
    with np.errstate(all="ignore"):
        mm = np.sqrt(mat["k2"] * mat["k3"])
        k8 = RQ.k8(mat["k1"], mat["k2"], mat["k3"])
    dt = pd.DatetimeIndex(dd.ravel())
    mon = np.where(ok.ravel(), dt.year * 12 + dt.month, -1).reshape(N, m)
    return Panel(tick, axis, mat["rv_tplus"], mat["k1"], mm, k8, ok, dd, mon.astype(np.int64))


def check_closed(P: Panel) -> list:
    """Нарушения PIT: у каждой бумаги дата цели предыдущей строки ≤ день прогноза следующей (иначе скор ещё не закрыт). Пусто — ок."""
    bad = []
    for j, t in enumerate(P.tick):
        i = np.flatnonzero(P.pres[:, j])
        tau, d = P.axis.to_numpy()[i], P.dd[i, j]
        if (tau[:-1] > d[1:]).any():
            bad.append(t)
    return bad


def long(P: Panel, **cols) -> pd.DataFrame:
    """Матрицы (N, m) -> длинная таблица (ticker, date d, t_plus τ, <cols>) только по строкам с pres."""
    n, j = np.nonzero(P.pres)
    out = pd.DataFrame({"ticker": np.array(P.tick)[j], "date": P.dd[n, j], "t_plus": P.axis.to_numpy()[n]})
    for c, a in cols.items():
        out[c] = a[n, j]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
