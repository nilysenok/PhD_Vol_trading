"""15. Гэп через уровень: триггер по low/high, выход на том же баре по close (не уровень, не open).

Спецификация п. 5; open в движках не используется вовсе.

Пути: engine.lvl_run (контртренд/диапазон/B), engine_trend.trend_run (тренд A/B). Следствие
правила: при гэпе убыток больше расстояния до стопа, а при отскоке внутри бара — меньше.
"""
import inspect

import numpy as np

from _eng import bars, mutant, pnl, run_lvl, run_trend
from fastbt import engine, engine_trend

ON_CLOSE = [("low[u] <= sl", "close[u] <= sl", 1), ("high[u] >= sl", "close[u] >= sl", 1)]
NEXT_BAR = [("tx[nt] = u if closed else end", "tx[nt] = min(u + 1, end) if closed else end", 1)]


def _both(repl):
    def f(mp):
        mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, repl))
        mp.setattr(engine_trend, "trend_run", mutant(engine_trend.trend_run, repl))
    return f


MUTATIONS = {"stop_trigger_on_close": _both(ON_CLOSE),     # уровень сравнивается с close, не с low/high
             "gap_exit_next_bar": _both(NEXT_BAR)}         # исполнение гэпа на следующем баре

N = 8


def _gap(low, high, close):
    c, h, lo = bars(np.full(N, 100.0))
    c = c.copy()
    lo[3], h[3], c[3] = low, high, close
    return c, h, lo


def test_engines_take_no_open():
    for f in (engine.lvl_run, engine_trend.trend_run):
        assert "open" not in inspect.signature(getattr(f, "py_func", f)).parameters


def test_gap_down_through_stop_recovers_inside_bar():
    """open 97 < sl 99, low 96,8, close 99,5: стоп сработал, выход по 99,5 (убыток меньше стопа)."""
    c, h, lo = _gap(96.8, 99.6, 99.5)
    tr = run_lvl(c, h, lo, longs=[1], a1=1, a2=5)
    assert tr == [(1, 3, 1)]
    assert np.isclose(pnl(tr, c)[0], np.log(99.5 / 100))


def test_gap_down_close_below_stop():
    """Закрытие 97,2 за стопом 99: убыток ln(0,972), а не ln(0,99) (уровень) и не ln(0,97) (open)."""
    c, h, lo = _gap(96.8, 97.6, 97.2)
    tr = run_lvl(c, h, lo, longs=[1], a1=1, a2=5)
    assert tr == [(1, 3, 1)]
    got = pnl(tr, c)[0]
    assert np.isclose(got, np.log(0.972))
    assert not np.isclose(got, np.log(0.99)) and not np.isclose(got, np.log(0.97))


def test_gap_up_through_take():
    c, h, lo = _gap(103.5, 104.5, 104.0)
    tr = run_lvl(c, h, lo, longs=[1], a1=1, a2=2)
    assert tr == [(1, 3, 1)]
    assert np.isclose(pnl(tr, c)[0], np.log(1.04))            # больше тейка 2%


def test_gap_up_through_short_stop():
    c, h, lo = _gap(102.0, 103.0, 102.5)
    tr = run_lvl(c, h, lo, shorts=[1], a1=1, a2=5, cd=2)
    assert tr == [(1, 3, -1)]
    assert np.isclose(pnl(tr, c)[0], -np.log(1.025))


def test_trend_gap_through_stop():
    c, h, lo = _gap(96.8, 99.6, 99.5)
    tr = run_trend(c, h, lo, longs=[1], isl=1.0)
    assert tr == [(1, 3, 1)]
    assert np.isclose(pnl(tr, c)[0], np.log(0.995))
    c, h, lo = _gap(100.5, 101.6, 101.5)
    tr = run_trend(c, h, lo, shorts=[1], isl=1.0)
    assert tr == [(1, 3, -1)]
    assert np.isclose(pnl(tr, c)[0], -np.log(1.015))
