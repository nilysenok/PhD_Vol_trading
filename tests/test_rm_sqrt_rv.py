"""E1в: выходы в единицах реализованной √RV (fastbt/rm_sqrt_rv.py). Синтетика с известной σ + мутации; реальные данные 2018 — только дым.
Мутации (в тестах, не в коде): RV с текущим днём; σ дня d+1 в уровне. Запуск: FASTBT_MAIN=<своя копия> pytest tests/test_rm_sqrt_rv.py
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import rm_sqrt_rv as rms
from fastbt import sweep_post as sp
from fastbt.batch import trades_one
from fastbt.engine import next_signal

N, E = 120, 30
DATES = pd.bdate_range("2018-01-01", periods=N).values.astype("datetime64[ns]")
ORIG = rms.sigma_from_rv


# ---------- σ: формула и отсутствие заглядывания ----------
def sigma_with_current(bar_dates, rv_dates, rv, w):
    """Мутация: окно оканчивается днём бара (текущий день входит)."""
    m = pd.Series(rv).rolling(w, min_periods=w).mean().to_numpy()
    j = np.searchsorted(rv_dates, bar_dates, side="right") - 1
    return np.where(j >= 0, np.sqrt(m[np.maximum(j, 0)]), np.nan)


def sigma_next_day(bar_dates, rv_dates, rv, w):
    """Мутация: σ дня d+1 приписана дню d."""
    s = ORIG(bar_dates, rv_dates, rv, w)
    return np.r_[s[1:], np.nan]


def ok_no_lookahead(fn) -> bool:
    """RV дня d меняет σ строго после d: σ_d не зависит от RV[d] и позже, σ_{d+1} видит RV[d]."""
    rng = np.random.default_rng(3)
    rv = rng.uniform(1e-4, 9e-4, N)
    d = 60
    pert = rv.copy()
    pert[d:] *= 25.0
    a, b = fn(DATES, DATES, rv, 5), fn(DATES, DATES, pert, 5)
    return bool(np.array_equal(a[: d + 1], b[: d + 1], equal_nan=True) and a[d + 1] != b[d + 1])


def test_sigma_formula_okno_do_d_minus_1():
    rv = np.arange(1.0, N + 1) * 1e-4
    s = rms.sigma_from_rv(DATES, DATES, rv, 5)
    assert np.isnan(s[:5]).all()
    assert s[5] == pytest.approx(np.sqrt(rv[0:5].mean())) and s[40] == pytest.approx(np.sqrt(rv[35:40].mean()))


def test_sigma_ne_vidit_tekushchiy_den():
    assert ok_no_lookahead(rms.sigma_from_rv)


def test_mutatsiya_rv_s_tekushchim_dnem_lovitsya():
    assert not ok_no_lookahead(sigma_with_current)


def test_mutatsiya_sigma_d_plus_1_lovitsya():
    assert not ok_no_lookahead(sigma_next_day)


def test_dyra_v_rv_ne_zapolnyaetsya():
    rd = np.r_[DATES[:20], DATES[60:]]
    s = rms.sigma_from_rv(DATES, rd, np.full(len(rd), 4e-4), 5)
    assert np.isnan(s[30:60]).all() and np.isfinite(s[:5]).sum() == 0 and np.isfinite(s[65])


# ---------- синтетика: уровни = k·σ ----------
def make_unit(trend: bool, lows: dict, highs: dict | None = None):
    n = N
    close = np.full(n, 100.0)
    high, low = np.full(n, 100.0), np.full(n, 100.0)
    for i, x in lows.items():
        low[i] = x
    for i, x in (highs or {}).items():
        high[i] = x
    sig = np.zeros(n, np.int8)
    sig[E] = 1
    z = np.zeros(n, bool)
    dm = np.full((1, 1, n), 0.1)
    return SimpleNamespace(
        ticker="X", dates=DATES, sigs=sig[None, :], nxt=np.stack([next_signal(sig)]), cat="Trend" if trend else "Contrarian",
        trend=trend, ind={}, high=high, low=low, close=close, n=n, start=0, exl=z[None, :], exs=z[None, :],
        adx=np.full(n, np.nan), atr=np.full(n, 1.0), atr0=np.full(n, 1.0), sdl=dm, sds=dm, tdl=dm, tds=dm,
        sig3=np.full((1, n), np.nan), med3=np.zeros(1), rh=np.zeros((1, n)), rl=np.zeros((1, n)))


def rv_step(j_big: int | None = None, big: float = 16e-4):
    rv = np.full(N, 4e-4)                              # σ = 0,02
    if j_big is not None:
        rv[j_big] = big                                # σ = 0,04 в окне, где этот день есть
    return DATES, rv


def rm_of(sid, **kw):
    r = dict(sp.RM_DEF["trend" if sid in ("S3", "S4") else "atr"], **rms.defaults(sid))
    r.update(kw)
    return r


def run(u, rm, rv):
    rms.apply(u, rm, rv)
    te, tx, td = trades_one(u, 0, u.rmP[0], u.trend, end=u.n)
    return te, tx, td


@pytest.mark.parametrize("lowx,exit_at", [(96.5, None), (95.9, E + 2)])
def test_stop_urovnya_ravny_k_sl_sigma_lvl(lowx, exit_at):
    u = make_unit(False, {E + 2: lowx})
    te, tx, _ = run(u, rm_of("S1", k_sl=2.0, okno_rv=5, max_hold=50), rv_step())      # стоп 100·(1−2·0,02) = 96
    assert te[0] == E and tx[0] == (exit_at if exit_at else E + 50)


def test_teyk_uroven_k_sl_ratio_sigma():
    u = make_unit(False, {}, {E + 3: 108.5})
    _, tx, _ = run(u, rm_of("S1", k_sl=2.0, ratio_tp_sl=2.0, okno_rv=5, max_hold=50), rv_step())   # тейк 100·(1+2·2·0,02) = 108
    assert tx[0] == E + 3
    u = make_unit(False, {}, {E + 3: 107.5})
    _, tx, _ = run(u, rm_of("S1", k_sl=2.0, ratio_tp_sl=2.0, okno_rv=5, max_hold=50), rv_step())
    assert tx[0] == E + 50


@pytest.mark.parametrize("lowx,exit_at", [(96.5, 120), (95.9, E + 2)])
def test_stop_trend_k_sl_sigma_bez_vremennogo(lowx, exit_at):
    u = make_unit(True, {E + 2: lowx})
    _, tx, _ = run(u, rm_of("S3", k_sl=2.0, k_trail=3.0, okno_rv=5, trail_n=10), rv_step())
    assert tx[0] == exit_at                              # exit_at=120=N: сделка открыта на конце (временного стопа нет)


def test_trend_gamma_vklyuchaet_vremennoy_stop_30():
    u = make_unit(True, {})
    _, tx, _ = run(u, rm_of("S3", k_sl=2.0, k_trail=3.0, okno_rv=5, trail_n=10, gamma_hold=1.0), rv_step())
    assert tx[0] == E + 30


def test_trejl_trenda_k_trail_sigma():
    u = make_unit(True, {E + 1: 114.5, E + 2: 113.9}, {E + 1: 120.0})   # макс.(окно) = 120 -> трейл 120 − 3·0,02·100 = 114
    _, tx, _ = run(u, rm_of("S3", k_sl=2.0, k_trail=3.0, okno_rv=5, trail_n=10), rv_step())
    assert tx[0] == E + 2                                # low 114,5 > 114 — держим; low 113,9 <= 114 — выход
    u = make_unit(True, {E + 1: 114.5, E + 2: 113.9}, {E + 1: 120.0})
    _, tx, _ = run(u, rm_of("S3", k_sl=2.0, k_trail=2.0, okno_rv=5, trail_n=10), rv_step())
    assert tx[0] == E + 1                                # k_trail=2: трейл 116 >= low 114,5 — выход на баре E+1



def test_bezubytok_v_sigma():
    u = make_unit(False, {E + 1: 100.5, E + 2: 100.5, E + 3: 100.5, E + 4: 99.9}, {E + 2: 106.0})
    u.close[E + 2] = 100.0 + 0.02 * 100 * 2.0            # ход 2σ > k_be = 1σ -> стоп на цену входа
    _, tx, _ = run(u, rm_of("S1", k_sl=2.0, k_be=1.0, okno_rv=5, max_hold=50), rv_step())
    assert tx[0] == E + 4


@pytest.mark.parametrize("mut", ["tekushchiy", "d_plus_1"])
def test_uroven_na_sigma_vhoda_mutatsii_lovyatsya(monkeypatch, mut):
    rv = rv_step(E, 16e-4)                               # RV дня входа вдвое выше по σ: у честной σ бара E его нет (okno_rv=1)
    low = {E + 1: 94.0}                                  # честный стоп 96 -> выход E+1; мутации дают стоп 92 -> нет выхода
    rm = rm_of("S1", k_sl=2.0, okno_rv=1, max_hold=50)
    _, tx, _ = run(make_unit(False, low), rm, rv)
    assert tx[0] == E + 1
    monkeypatch.setattr(rms, "sigma_from_rv", sigma_with_current if mut == "tekushchiy" else sigma_next_day)
    _, tx, _ = run(make_unit(False, low), rm, rv)
    assert tx[0] != E + 1


def test_net_sigmy_net_vhoda():
    u = make_unit(False, {})
    te, _, _ = run(u, rm_of("S1", okno_rv=60), (DATES[:10], np.full(10, 4e-4)))     # истории RV нет -> σ NaN
    assert len(te) == 0

