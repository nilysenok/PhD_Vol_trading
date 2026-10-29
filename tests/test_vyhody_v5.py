"""Выходы сетки v5 (Ж-127): Chandelier и ATR-trailing тренда S3/S4 (trend_run, kind 0).

Chandelier (tt = 0, 1 в движке — одна формула SW.bt_trend_v3): tsl = max(high, N баров, с текущим и ДО входа) − k·ATR[t].
ATR-trailing (tt = 3, новый): tsl = max(high с входа, с текущим баром) − k·ATR[t]; храповик sl = max(sl, tsl).
Выход по low ≤ sl на том же баре, исполнение по close бара (спецификация, tests/audit_fastbt/test_15_gap.py).
Быстрый слой (numba) сверяется с независимой медленной реализацией fastbt.vyhody_v5_medl на фиксированном наборе точек.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "audit_fastbt"))
from _eng import bars, mutant, run_trend  # noqa: E402
from fastbt import engine, engine_trend  # noqa: E402
from fastbt.vyhody_v5_medl import trend_medl  # noqa: E402

N = 10
TT_ATR = 3


def _path(rows):
    c, h, lo = bars(np.full(N, 100.0))
    c, h, lo = c.copy(), h.copy(), lo.copy()
    for t, (cc, hh, ll) in rows.items():
        c[t], h[t], lo[t] = cc, hh, ll
    return c, h, lo


# бар 0 (до входа) — максимум 110; вход на баре 1 по 100, ATR = 1, начальный стоп 98, k = 2
PRE = {0: (100.0, 110.0, 99.5)}
RISE = {2: (101.0, 101.5, 100.5), 3: (101.0, 101.0, 99.6), 4: (100.0, 100.5, 99.4)}


def test_chandelier_uses_prior_high_exits_at_once():
    c, h, lo = _path({**PRE, **RISE})
    # rh = 110 (окно 11 баров с баром 0) → tsl = 108 > low бара 2 → выход на баре 2
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=1) == [(1, 2, 1)]


def test_atr_trail_ignores_prior_high_ratchets_exits_bar4():
    c, h, lo = _path({**PRE, **RISE})
    # ext с входа: бар 2 → 101,5, tsl = 99,5; бар 3 low 99,6 > 99,5; бар 4 low 99,4 ≤ 99,5 → выход
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR) == [(1, 4, 1)]


def test_atr_trail_ratchet_holds_when_atr_widens():
    c, h, lo = _path({**PRE, **RISE})
    atr = np.ones(N)
    atr[3:] = 3.0       # tsl = 101,5 − 6 = 95,5, храповик держит 99,5
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR, atr=atr) == [(1, 4, 1)]


def test_atr_trail_same_bar_high_counts():
    c, h, lo = _path({2: (102.0, 104.0, 101.5)})
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR) == [(1, 2, 1)]   # 104 − 2 = 102 ≥ 101,5


def test_atr_trail_short_mirror():
    c, h, lo = _path({2: (99.0, 99.5, 98.5), 3: (99.0, 99.4, 99.0), 4: (100.0, 100.6, 99.5)})
    # ext = 98,5 → tsl = 100,5; бар 4 high 100,6 ≥ 100,5 → выход
    assert run_trend(c, h, lo, shorts=[1], isl=2.0, tmult=2.0, tt=TT_ATR) == [(1, 4, -1)]


def test_atr_trail_gap_through_stop_exits_on_gap_bar_at_close():
    c, h, lo = _path({2: (101.0, 101.5, 100.5), 3: (95.0, 96.0, 94.0), 4: (99.0, 99.5, 98.5)})
    tr = run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR)
    assert tr == [(1, 3, 1)]
    assert np.log(c[3] / c[1]) == pytest.approx(np.log(0.95))      # исполнение по close гэп-бара, не по уровню


def test_default_tt_unchanged_fixed_equals_chandelier():
    c, h, lo = _path({**PRE, **RISE})
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=0) == run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=1)


# --- мутации № 2, 3, 4 для нового режима ---
def _mut(repl):
    return lambda mp: mp.setattr(engine_trend, "trend_run", mutant(engine_trend.trend_run, repl))


def _run_path(rows, **kw):
    c, h, lo = _path(rows)
    return run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR, **kw)


def test_mutatsiya_2_kasanie_urovnya_lovitsya(monkeypatch):
    # low ровно на стопе: 101,5 − 2 = 99,5; low 99,5 ≤ 99,5 — выход (касание = срабатывание стопа)
    rows = {2: (101.0, 101.5, 100.5), 3: (100.0, 100.5, 99.5)}
    assert _run_path(rows) == [(1, 3, 1)]
    _mut([("low[u] <= sl", "low[u] < sl", 1)])(monkeypatch)
    assert _run_path(rows) != [(1, 3, 1)]


def test_mutatsiya_3_poryadok_sobytij_lovitsya(monkeypatch):
    # стоп и сигнал выхода на одном баре: приоритет у стопа (cooldown только после стопа)
    rows = {2: (101.0, 101.5, 100.5), 3: (100.0, 100.5, 99.0)}
    kw = dict(cd=3)
    c, h, lo = _path(rows)
    base = run_trend(c, h, lo, longs=[1, 4], exl=[3], isl=2.0, tmult=2.0, tt=TT_ATR, **kw)
    assert base == [(1, 3, 1)]                      # вход на 4 заблокирован cooldown после стопа
    swap = [("if low[u] <= sl:\n                    closed = True\n                    stop = True\n                elif exl[u]:\n                    closed = True",
             "if exl[u]:\n                    closed = True\n                elif low[u] <= sl:\n                    closed = True\n                    stop = True", 1)]
    _mut(swap)(monkeypatch)
    assert run_trend(c, h, lo, longs=[1, 4], exl=[3], isl=2.0, tmult=2.0, tt=TT_ATR, **kw) != base


def test_mutatsiya_4_gep_po_close_lovitsya(monkeypatch):
    rows = {2: (101.0, 101.5, 100.5), 3: (101.0, 101.2, 94.0)}      # гэп-проход внутри бара, close выше стопа
    assert _run_path(rows) == [(1, 3, 1)]
    _mut([("low[u] <= sl", "close[u] <= sl", 1)])(monkeypatch)
    assert _run_path(rows) != [(1, 3, 1)]


def test_mutatsiya_atr_trail_kak_chandelier_lovitsya(monkeypatch):
    c, h, lo = _path({**PRE, **RISE})
    _mut([("tt == 3", "tt == 99", 2)])(monkeypatch)       # ветка atr_trail отключена → прежний путь
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=TT_ATR) != [(1, 4, 1)]


# --- быстрый = медленный (proverka.md п. 3) ---
def _points(seed, n=60):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    high = close * (1 + np.abs(r.normal(0, 0.005, n)))
    low = close * (1 - np.abs(r.normal(0, 0.005, n)))
    atr = np.full(n, 1.0) * (0.5 + r.random(n))
    sig = np.zeros(n, int)
    sig[r.random(n) < 0.12] = 1
    sig[r.random(n) < 0.08] = -1
    return close, high, low, atr, sig


@pytest.mark.parametrize("tt", [0, 1, 2, TT_ATR])
@pytest.mark.parametrize("seed", range(25))
def test_bystryj_raven_medlennomu(seed, tt):
    close, high, low, atr, sig = _points(seed)
    kw = dict(isl=2.0, tmult=1.5 + (seed % 3), bth=1.0 if seed % 2 else 1e12, tt=tt, cd=seed % 3)
    longs, shorts = list(np.flatnonzero(sig == 1)), list(np.flatnonzero(sig == -1))
    fast = run_trend(close, high, low, longs=longs, shorts=shorts, atr=atr, **kw)
    ml = trend_medl(close, high, low, atr, sig, **kw)
    assert fast == ml and (len(fast) > 0 or seed > 3)
