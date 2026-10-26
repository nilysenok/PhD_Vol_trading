"""14. Стоп и тейк в одном баре: приоритет у стопа, cooldown только после стопа; выход по ADX.

Выход по close бара; ближайший вход после стопа — бар выхода + cd (спецификация пп. 3, 4, 14).

Путь: engine.lvl_run (контртренд/диапазон, A: ATR-уровни, S5/S6: расстояния, B: σ-уровни).
Стоп и тейк — триггеры по low/high; цена выхода одинакова (close), поэтому приоритет виден по
cooldown: после стопа вход заблокирован на cd баров.
"""
import numpy as np
import pytest

from _eng import bars, mutant, pnl, run_lvl
from fastbt import engine

I16, I20 = " " * 16, " " * 20


def _blk(first, second, stop_first):
    """Тело проверки уровней long/short в исходнике lvl_run."""
    a, b = (first, second) if stop_first else (second, first)
    s = lambda cond, stop: f"{cond}:\n{I20}closed = True" + (f"\n{I20}stop = True" if stop else "")
    return f"if {s(a, stop_first)}\n{I16}elif {s(b, not stop_first)}"


TP_FIRST = [(_blk("low[u] <= sl", "high[u] >= tp", True), _blk("low[u] <= sl", "high[u] >= tp", False), 1),
            (_blk("high[u] >= sl", "low[u] <= tp", True), _blk("high[u] >= sl", "low[u] <= tp", False), 1)]

MUTATIONS = {
    "tp_before_sl": lambda mp: mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, TP_FIRST)),
    "cooldown_after_tp": lambda mp: mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, [
        ("t = u + cd if (stop and cd > 0) else u", "t = u + cd if cd > 0 else u", 1)])),
    "range_adx_exit_off": lambda mp: mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, [
        ("if not closed and adx_on > 0:", "if not closed and adx_on > 1:", 1)])),
}

N = 10
CL = np.full(N, 100.0)


def _bar3(high, low, close=100.5):
    c, h, lo = bars(CL)
    c = c.copy()
    h[3], lo[3], c[3] = high, low, close
    return c, h, lo


# уровни входа на баре 1 (close 100): long sl 99 / tp 102, short sl 101 / tp 98
KINDS = {"atr": dict(kind=0, a1=1, a2=2, atr=1.0),
         "dist": dict(kind=1, sd=np.full(N, 1.0), tdist=np.full(N, 2.0)),
         "sigma": dict(kind=2, a1=1, a2=2, sigma=np.full(N, 0.01))}


@pytest.mark.parametrize("k", list(KINDS))
def test_both_levels_in_bar_long_stop_wins(k):
    c, h, lo = _bar3(102.5, 98.5)
    tr = run_lvl(c, h, lo, longs=[1, 4, 6], cd=3, **KINDS[k])
    # стоп на баре 3 → cooldown 3 → бары 3..5 заблокированы, вход на 6 (сигнал 4 пропущен)
    assert tr == [(1, 3, 1), (6, N, 1)]
    assert np.isclose(pnl(tr[:1], c)[0], np.log(100.5 / 100))   # выход по close, не по уровню


@pytest.mark.parametrize("k", list(KINDS))
def test_both_levels_in_bar_short_stop_wins(k):
    c, h, lo = _bar3(101.5, 97.5, 99.0)
    tr = run_lvl(c, h, lo, shorts=[1, 4, 6], cd=3, **KINDS[k])
    assert tr == [(1, 3, -1), (6, N, -1)]
    assert np.isclose(pnl(tr[:1], c)[0], -np.log(99.0 / 100))


def test_take_only_no_cooldown():
    c, h, lo = _bar3(102.5, 99.5)
    assert run_lvl(c, h, lo, longs=[1, 4, 6], cd=3, a1=1, a2=2) == [(1, 3, 1), (4, N, 1)]


def test_stop_only_cooldown():
    c, h, lo = _bar3(100.5, 98.5)
    assert run_lvl(c, h, lo, longs=[1, 4, 6], cd=3, a1=1, a2=2) == [(1, 3, 1), (6, N, 1)]


def test_reentry_on_exit_bar_after_take():
    """Повторный вход на баре выхода (не после стопа) — спецификация п. 14."""
    c, h, lo = _bar3(102.5, 99.5, 100.0)
    assert run_lvl(c, h, lo, longs=[1, 3], cd=3, a1=1, a2=2) == [(1, 3, 1), (3, N, 1)]


def test_range_adx_exit():
    """Диапазон (S5/S6, B-диапазон): выход по ADX14 > 30 после стопа/тейка, не стоповый."""
    c, h, lo = bars(CL)
    adx = np.full(N, 20.0)
    adx[4] = 30.0
    adx[5] = 30.5
    tr = run_lvl(c, h, lo, longs=[1, 5], a1=5, a2=5, adx_on=1, adx=adx, cd=3)
    assert tr == [(1, 5, 1), (5, N, 1)]           # вход на 5 сразу (не стоп), ADX 30 — не выход
    assert run_lvl(c, h, lo, longs=[1], a1=5, a2=5, adx_on=0, adx=adx) == [(1, N, 1)]


def test_levels_fixed_at_entry_atr():
    """ATR берётся на баре входа: рост ATR после входа уровни не двигает."""
    c, h, lo = _bar3(100.5, 98.5)
    atr = np.full(N, 1.0)
    atr[2:] = 5.0
    assert run_lvl(c, h, lo, longs=[1], a1=1, a2=2, atr=atr) == [(1, 3, 1)]
