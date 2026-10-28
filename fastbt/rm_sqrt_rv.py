"""Выходы в единицах реализованной σ = √RV (Ж-031 Э1-в п. 4, блок `rm_sqrt_rv` сеток Э1-б): стоп, тейк, трейл, безубыток, удержание.

Включается только при `rm_tip=sqrt_rv` (S5, S6) или `rm_edinitsa=sqrt_rv` (S1–S4); пути ATR/pivot/vwap не затронуты.
Ядро fastbt не правится: используется готовый kind=2 (`engine.lvl_run`, `engine_trend.trend_run`) со слотом σ = `sig3`.
 - σ бара t = √(среднее RV за okno_rv дней, оканчивающихся днём t−1): RV дня t и позже в σ не входит;
 - стоп = цена·(1 ∓ k_sl·σ), тейк = цена·(1 ± k_sl·ratio_tp_sl·σ), трейл тренда = макс.(trail_n) − k_trail·σ[t]·цена,
   безубыток после хода k_be·σ·цена; удержание max_hold, а при gamma_hold — max_hold·(σ̄/σ)^γ в границах [3, 3·max_hold];
   у тренда базы нет: gamma_hold=null — без временного стопа (как A), иначе база 30 дней;
 - σ̄ — медиана σ по барам до 2018-01-01 (в 2018–2021 не пересчитывается: без заглядывания вперёд);
 - нет σ на баре входа (мало истории RV) — входа нет, константа 0,02 из ядра не подставляется (Д4).
Бары-оценки не копируются: apply() работает на копии Unit из sweep_post.build_view.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fastbt.batch import (L_ADX, L_A1, L_A2, L_BE, L_CD, L_DIST, L_GAM, L_KIND, L_MH, L_SIG, L_TD,
                          T_BASE, T_BTH, T_CD, T_GAM, T_ISL, T_KIND, T_RH, T_SIG, T_TM)
from fastbt.engine import next_signal
from fastbt.indicators import rolling_hl

PRE = np.datetime64("2018-01-01")        # σ̄ (медиана) — только по барам до начала настройки
TREND_BASE, NO_HOLD = 30.0, 1e9          # база временного стопа тренда при gamma_hold; «без временного стопа»
STALE_DAYS = 10                          # последний день RV старше бара больше, чем на столько дней — σ нет
_K = {"S1": (2.0, 2.0), "S2": (2.0, 2.0), "S3": (3.0, 2.0), "S4": (3.0, 2.0), "S5": (0.75, 1.0), "S6": (1.25, 1.0)}


def defaults(sid: str) -> dict:
    """Исходные значения блока (= `ishodnoe` сеток Э1-б; сверяется тестом). max_hold у S5/S6 — 8, у S1/S2 — 20 из RM_DEF."""
    d = dict(k_sl=_K[sid][0], ratio_tp_sl=_K[sid][1], k_trail=3.0, k_be=None, gamma_hold=None, okno_rv=20)
    return {**d, "max_hold": 8} if sid in ("S5", "S6") else d


@lru_cache(maxsize=1)
def _rv_all() -> pd.DataFrame:
    from fastbt import sigma_e
    return sigma_e.rv_dev()              # дневной RV, только ≤ 2021-12-31 (фильтр чтения + assert)


def rv_ticker(ticker: str) -> tuple:
    d = _rv_all()
    x = d[d["ticker"] == ticker].sort_values("date")
    return x["date"].values.astype("datetime64[ns]"), x["rv"].values.astype(np.float64)


def sigma_from_rv(bar_dates, rv_dates, rv, w: int) -> np.ndarray:
    """σ[t] = √(среднее RV по w последним строкам с датой СТРОГО раньше даты бара t); NaN — мало истории / дыра > STALE_DAYS."""
    m = pd.Series(rv).rolling(int(w), min_periods=int(w)).mean().to_numpy()      # окно оканчивается строкой j (включительно)
    bd = np.asarray(bar_dates, "datetime64[ns]")
    j = np.searchsorted(np.asarray(rv_dates, "datetime64[ns]"), bd, side="left") - 1   # последняя строка с датой < даты бара
    out = np.full(len(bd), np.nan)
    ok = j >= 0
    ok[ok] &= (bd[ok] - np.asarray(rv_dates, "datetime64[ns]")[j[ok]]) <= np.timedelta64(STALE_DAYS, "D")
    out[ok] = np.sqrt(m[j[ok]])
    return out


def sigma_median(sigma: np.ndarray, dates) -> float:
    s = sigma[(np.asarray(dates, "datetime64[ns]") < PRE) & np.isfinite(sigma)]
    return float(np.median(s)) if len(s) else 0.0


def _rows(rm: dict, trend: bool, adx_on: int) -> np.ndarray:
    g, be = float(rm["gamma_hold"] or 0.0), rm["k_be"]
    if trend:
        P = np.zeros((1, 12))
        P[0, [T_KIND, T_ISL, T_TM, T_BTH, T_CD, T_RH, T_BASE, T_GAM, T_SIG]] = [
            2, rm["k_sl"], rm["k_trail"], be or 0.0, rm["cooldown_bars"], 0, TREND_BASE if g > 0 else NO_HOLD, g, 0]
        return P
    P = np.zeros((1, 11))
    P[0, [L_KIND, L_A1, L_A2, L_MH, L_BE, L_CD, L_TD, L_ADX, L_DIST, L_GAM, L_SIG]] = [
        2, rm["k_sl"], rm["k_sl"] * rm["ratio_tp_sl"], rm["max_hold"], be or -1.0, rm["cooldown_bars"], 0, adx_on, 0, g, 0]
    return P


def _trail_hl(v, tn: int) -> tuple:
    if f"rh_{tn}" in v.ind:
        return v.ind[f"rh_{tn}"], v.ind[f"rl_{tn}"]
    return rolling_hl(v.high, v.low, tn)


def apply(v, rm: dict, rv: tuple | None = None) -> None:
    """Копия Unit v -> выходы в σ: строка RM kind=2, ряд σ, σ̄, сигналы без входов там, где σ нет. rv — (даты, RV) для тестов."""
    rd, rr = rv if rv is not None else rv_ticker(v.ticker)
    sg = sigma_from_rv(v.dates, rd, rr, int(rm["okno_rv"]))
    s = np.where(np.isnan(sg), 0, v.sigs[0]).astype(np.int8)
    v.sigs = np.ascontiguousarray(s[None, :])
    v.nxt = np.stack([next_signal(x) for x in v.sigs])
    v.sig3 = np.ascontiguousarray(sg[None, :])
    v.med3 = np.array([sigma_median(sg, v.dates)])
    adx_on = 0 if v.cat == "Contrarian" else 1
    v.rmP = _rows(rm, bool(v.trend), adx_on)
    if v.trend:
        rh, rl = _trail_hl(v, int(rm["trail_n"]))
        v.rh, v.rl = np.ascontiguousarray(rh[None, :]), np.ascontiguousarray(rl[None, :])
