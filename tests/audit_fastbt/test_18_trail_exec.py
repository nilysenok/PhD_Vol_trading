"""18. Трейлинг тренда (храповик, parabolic) и слой исполнения (consec, min_hold, cooldown).

Трейл: tsl = rh_N[t] − k·ATR[t] (rh включает high[t]),
храповик sl = max(sl, tsl); parabolic: шаг растёт на ps при новом экстремуме до pm. Слой
исполнения: сила → |pos| > 0,5 → consec_entry → min_hold → cooldown → знак сырой позиции
(спецификация пп. 3, 13, 18.9).

Пути: engine_trend.trend_run, execlayer.exec_layer.
"""
import numpy as np

from _eng import bars, mutant, run_trend
from fastbt import engine_trend, execlayer


def _tr(repl):
    return lambda mp: mp.setattr(engine_trend, "trend_run", mutant(engine_trend.trend_run, repl))


def _ex(repl):
    return lambda mp: mp.setattr(execlayer, "exec_layer", mutant(execlayer.exec_layer, repl))


MUTATIONS = {
    "trail_no_ratchet": _tr([("sl = max(sl, tsl)", "sl = tsl", 1), ("sl = min(sl, tsl)", "sl = tsl", 1)]),
    "trail_rh_prev_bar": _tr([("tsl = rh[u] - tmult * ca", "tsl = rh[u - 1] - tmult * ca", 1)]),
    "parabolic_no_accel": _tr([("step = min(step + ps, pm)", "step = step", 2)]),
    "exec_cooldown_short": _ex([("cl = cool", "cl = cool - 1", 1)]),
    "exec_consec_off1": _ex([("cnt >= consec", "cnt >= consec - 1", 1)]),
    # min_hold продлевает позицию за пределы сырой сделки (знак прошлого бара)
    "exec_min_hold_extends": _ex([("out[i] = b[i] * s",
                                   "out[i] = b[i] * (s if s != 0.0 else (out[i - 1] if i > 0 else 0.0))",
                                   1)]),
}

N = 10


def _path(rows):
    c, h, lo = bars(np.full(N, 100.0))
    c, h, lo = c.copy(), h.copy(), lo.copy()
    for t, (cc, hh, ll) in rows.items():
        c[t], h[t], lo[t] = cc, hh, ll
    return c, h, lo


def test_trail_ratchet_holds_when_atr_widens():
    c, h, lo = _path({2: (102.5, 103.0, 102.0), 3: (101.0, 102.0, 100.5)})
    atr = np.ones(N)
    atr[3:] = 3.0
    # бар 2: tsl = 103 − 2·1 = 101 → sl 101; бар 3: tsl = 103 − 6 = 97, храповик держит 101
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, atr=atr) == [(1, 3, 1)]
    cs, hs, ls = _path({2: (97.5, 98.0, 97.0), 3: (99.0, 99.5, 98.0)})
    assert run_trend(cs, hs, ls, shorts=[1], isl=2.0, tmult=2.0, atr=atr) == [(1, 3, -1)]


def test_trail_uses_high_of_same_bar():
    """rh_N включает high[t]: новый максимум 104 и low 101,5 ≤ 104 − 2 на том же баре — выход."""
    c, h, lo = _path({2: (102.0, 104.0, 101.5)})
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0) == [(1, 2, 1)]


def test_parabolic_by_hand():
    # sl 98, ext 100,5, шаг 0,02; бар 2: ext 102, шаг 0,04 → tsl 98,16;
    # бар 3: tsl = 98,16 + 0,04·(102 − 98,16) = 98,3136 ≥ low 98,2 → выход
    c, h, lo = _path({2: (101.5, 102.0, 101.0), 3: (98.5, 101.0, 98.2)})
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tt=2, ps=0.02, pm=0.2) == [(1, 3, 1)]


RAW = np.array([0, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0], float)
ONE = np.ones(len(RAW))


def _el(consec=1, hold=0, cool=0, mstr=0.0, st=ONE, end=len(RAW)):
    return execlayer.exec_layer(RAW, st, consec, hold, cool, mstr, end)


def test_exec_trivial_and_end():
    assert np.array_equal(_el(), RAW)
    assert np.array_equal(_el(end=8), np.r_[RAW[:8], np.zeros(4)])


def test_exec_consec_entry():
    assert np.array_equal(_el(consec=2), [0, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0])
    assert np.array_equal(_el(consec=3), [0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0])


def test_exec_cooldown():
    # переход 1→0 на баре 5 → бары 6, 7 принудительно 0
    assert np.array_equal(_el(cool=2), [0, 1, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0])


def test_exec_strength_and_min_hold():
    st = ONE.copy()
    st[3] = 0.2
    assert np.array_equal(_el(mstr=0.5, st=st), [0, 1, 1, 0, 1, 0, 0, -1, -1, -1, 0, 0])
    # min_hold возвращает бар 3, но не продлевает сделку за бар 4 (знак сырой позиции)
    assert np.array_equal(_el(mstr=0.5, st=st, hold=3), RAW)
