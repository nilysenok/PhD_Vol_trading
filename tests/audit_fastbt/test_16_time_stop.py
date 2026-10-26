"""16. Временной стоп: выход ровно на баре вход + max_hold; динамический max_hold B.

held = 0 на баре входа, +1 на каждом следующем; выход по close. B: max_hold_t = int(clip(base·(σ_med/σ_вх)^γ, 3, 3·base)); тренд A
временного стопа не имеет, тренд B — имеет (base 30) (спецификация пп. 6, 15).

Пути: engine.lvl_run, engine._dyn_hold, engine_trend.trend_run.
"""
import numpy as np

from _eng import bars, mutant, run_lvl, run_trend
from fastbt import engine, engine_trend

LVL_TS = "if not closed and held >= mht:"
TR_TS = "kind == 2 and held >= mht"


def _ts(lvl_new, tr_new):
    def f(mp):
        mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, [(LVL_TS, lvl_new, 1)]))
        mp.setattr(engine_trend, "trend_run", mutant(engine_trend.trend_run, [(TR_TS, tr_new, 1)]))
    return f


def _dyn_noclip(base, sv, med, gamma):
    if gamma > 0.0 and sv > 1e-8 and med > 1e-8:
        return int(base * (med / sv) ** gamma)
    return int(base)


def _noclip(mp):
    mp.setattr(engine, "_dyn_hold", _dyn_noclip)
    mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, [], {"_dyn_hold": _dyn_noclip}))
    mp.setattr(engine_trend, "trend_run", mutant(engine_trend.trend_run, [],
                                                 {"_dyn_hold": _dyn_noclip}))


MUTATIONS = {
    "time_stop_plus1": _ts("if not closed and held > mht:", "kind == 2 and held > mht"),
    "time_stop_minus1": _ts("if not closed and held >= mht - 1:", "kind == 2 and held >= mht - 1"),
    "trend_a_time_stop": lambda mp: mp.setattr(engine_trend, "trend_run", mutant(
        engine_trend.trend_run, [(TR_TS, "held >= (mht if kind == 2 else 30)", 1)])),
    "dyn_hold_no_clip": _noclip,
    "time_decay_off": lambda mp: mp.setattr(engine, "lvl_run", mutant(
        engine.lvl_run, [("if tdec > 0 and held > mht // 2:", "if False:", 1)])),
}


def test_time_decay_halves_take_after_half_hold():
    """time_decay (S5/S6 ATR-варианты): после held > max_hold//2 тейк = ep ± 0,5·исходного."""
    c, h, lo = _flat(14)
    h[4] = h[7] = 102.5                       # тейк 104; с held 6 > 10//2 — 102
    assert run_lvl(c, h, lo, longs=[1], a1=50, a2=4, mh=10, tdec=1) == [(1, 7, 1)]
    assert run_lvl(c, h, lo, longs=[1], a1=50, a2=4, mh=10, tdec=0) == [(1, 11, 1)]
    lo2 = lo.copy()
    lo2[7] = 97.5
    assert run_lvl(c, h, lo2, shorts=[1], a1=50, a2=4, mh=10, tdec=1) == [(1, 7, -1)]


def _flat(n):
    return bars(np.full(n, 100.0))


def test_time_stop_exact_bar():
    c, h, lo = _flat(14)
    assert run_lvl(c, h, lo, longs=[2], a1=50, a2=50, mh=5) == [(2, 7, 1)]
    # после временного стопа cooldown нет: повторный вход на баре выхода
    assert run_lvl(c, h, lo, longs=[2, 7], a1=50, a2=50, mh=5, cd=3) == [(2, 7, 1), (7, 12, 1)]
    assert run_lvl(c, h, lo, shorts=[3], a1=50, a2=50, mh=1) == [(3, 4, -1)]


def test_signal_exit_before_time_stop():
    c, h, lo = _flat(14)
    assert run_lvl(c, h, lo, longs=[2], exl=[5], a1=50, a2=50, mh=5) == [(2, 5, 1)]


def test_dyn_hold_by_hand():
    f = engine._dyn_hold
    assert f(20.0, 0.02, 0.01, 1.0) == 10                 # 20·0,5
    assert f(20.0, 0.02, 0.01, 0.5) == 14                 # int(20·√0,5) = int(14,14)
    assert f(20.0, 0.2, 0.01, 1.0) == 3                   # 1 → нижняя граница 3
    assert f(20.0, 1e-4, 0.01, 1.0) == 60                 # 2000 → верхняя 3·base
    assert f(20.0, 0.02, 0.01, -1.0) == 20                # γ = None → base


def test_b_dynamic_time_stop_in_engine():
    c, h, lo = _flat(80)
    sg = np.full(80, 0.02)
    kw = dict(kind=2, a1=50, a2=50, mh=20, med=0.01)
    assert run_lvl(c, h, lo, longs=[1], sigma=sg, gamma=1.0, **kw) == [(1, 11, 1)]
    assert run_lvl(c, h, lo, longs=[1], sigma=sg, gamma=-1.0, **kw) == [(1, 21, 1)]
    sg2 = sg.copy()
    sg2[1] = 0.2
    assert run_lvl(c, h, lo, longs=[1], sigma=sg2, gamma=1.0, **kw) == [(1, 4, 1)]
    sg2[1] = 1e-4                                         # уровни ±k·σ·close — раздвинуть
    assert run_lvl(c, h, lo, longs=[1], sigma=sg2, gamma=1.0,
                   **{**kw, "a1": 1e5, "a2": 1e5}) == [(1, 61, 1)]
    sg2[1] = np.nan                                       # σ = NaN → 0,02 (SW:1218-1222)
    assert run_lvl(c, h, lo, longs=[1], sigma=sg2, gamma=1.0, **kw) == [(1, 11, 1)]


def test_trend_a_no_time_stop_trend_b_base30():
    c, h, lo = _flat(100)
    assert run_trend(c, h, lo, longs=[1], exl=[90], isl=50) == [(1, 90, 1)]
    assert run_trend(c, h, lo, longs=[1], exl=[90], isl=50, kind=2, tmult=50) == [(1, 31, 1)]
    sg = np.full(100, 0.02)
    assert run_trend(c, h, lo, shorts=[1], isl=50, kind=2, tmult=50, gamma=1.0, med=0.01,
                     sigma=sg) == [(1, 16, -1)]
