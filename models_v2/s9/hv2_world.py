"""Ж-214, лист 2: «мир» дневного ряда SBER для направлений V5/V6 — панель A0 (models_v2/har.py: ln RV_d, ln RV_w, ln RV_m, ln(1+J)), M10-величины RQ и rv_rel,
сетка 2271 дата (`har.eligible`). Все признаки строки t известны к 18:50 t. Пары обучения прогноза с i: (t, t+1), t+1 <= i, один сегмент, признаки конечны.
`corrupt` — порча всех строк > i0 (главный тест обрезки). `zscore` — z-оценка качества измерения по дням s < t (расширяющаяся, причинная)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import daily as D, har as H


@dataclass
class World:
    p: H.Panel
    rq: np.ndarray       # RQ дня (%^4), строка t
    rel: np.ndarray      # rv_rel = sqrt(2 RQ / n) / RV, строка t
    el: np.ndarray       # строки сетки (день прогноза d)
    mkt: np.ndarray | None = None   # rv_w IMOEX (среднее 5 дней, %²), строка t (H6)

    @property
    def d(self) -> np.ndarray:
        return self.p.dates[self.el]

    @property
    def td(self) -> np.ndarray:
        return self.p.dates[self.el + 1]

    @property
    def target(self) -> np.ndarray:
        return self.p.rv[self.el + 1]


def build(bars: pd.DataFrame, m10: pd.DataFrame, imoex: pd.DataFrame | None = None) -> World:
    d, _ = D.build_daily(bars)
    p = H.make_panel(d)
    m = m10.reindex(d.index)
    mkt = None if imoex is None else D.build_daily(imoex, index_days=None)[0]["rv_w"].reindex(d.index).to_numpy(float)
    return World(p, m["rq"].to_numpy(float), m["rv_rel"].to_numpy(float), H.eligible(p), mkt)


def sub(w: World, i0: int) -> World:
    """Сетка до строки i0 включительно (прогнозы на d <= i0)."""
    return dataclasses.replace(w, el=w.el[w.el <= i0])


def corrupt(w: World, i0: int, seed: int = 7) -> World:
    """Строки > i0 (RV, признаки панели, RQ, rv_rel) домножены/сдвинуты случайно; строки <= i0 без изменений."""
    rng = np.random.default_rng(seed)
    n = len(w.p.rv)
    m = np.ones(n)
    m[i0 + 1:] = np.exp(rng.normal(0, 1.0, n - i0 - 1))
    X = w.p.X + np.where(np.arange(n)[:, None] > i0, rng.normal(0, 1.0, w.p.X.shape), 0.0)
    p = H.Panel(w.p.dates, w.p.seg, w.p.rv * m, X)
    mkt = None if w.mkt is None else w.mkt * m ** 2
    return dataclasses.replace(w, p=p, rq=w.rq * m ** 2, rel=w.rel * np.exp(np.where(np.arange(n) > i0, rng.normal(0, 1.0, n), 0.0)), mkt=mkt)


def pairs_idx(seg: np.ndarray, F: np.ndarray, i: int) -> np.ndarray:
    """t с t+1 <= i, seg[t] == seg[t+1], признаки F[t] конечны."""
    t = np.arange(0, i)
    ok = (seg[t] == seg[t + 1]) & np.isfinite(F[t]).all(axis=1)
    return t[ok]


def zscore(x: np.ndarray, min_obs: int = 60, lead: int = 0, win: int | None = None) -> np.ndarray:
    """z_t = (x_t − mean_{s<t} x_s) / sd_{s<t} (ddof = 1; нужно >= min_obs конечных s < t; win — окно последних win дней вместо расширяющегося).
    `lead` > 0 — МУТАНТ «день d+1»: значение берётся из строки t + lead (должен ловиться тестом обрезки)."""
    x = np.asarray(x, float)
    n = len(x)
    xv = np.full(n, np.nan)
    xv[:n - lead] = x[lead:]
    out = np.full(n, np.nan)
    for t in range(n):
        past = x[max(0, t - win) if win else 0:t]
        past = past[np.isfinite(past)]
        if len(past) >= min_obs and past.std(ddof=1) > 0:
            out[t] = (xv[t] - past.mean()) / past.std(ddof=1)
    return out
