"""17. Безубыток: стоп на ep при (close − ep)·dir > порог, проверка до стопа того же бара.

При (close − ep)·dir > порог стоп переносится на ep; проверка по close бара — до
проверки стопа на том же баре; срабатывает один раз; выход по безубытку — стоповый (cooldown).
Порог: A — be·ATR_вх; S5/S6 — 0,5·расстояние тейка; B — k_be·σ[текущий бар]·ep; тренд A —
be·ATR_вх с ограничением трейла ценой входа; тренд B — k_be·σ_вх·ep (спецификация пп. 7, 14, 18.9).

Пути: engine.lvl_run, engine_trend.trend_run.
"""
import numpy as np

from _eng import bars, mutant, run_lvl, run_trend
from fastbt import engine, engine_trend

S8, S12, S16 = " " * 8, " " * 12, " " * 16
BE_LATE = [("be_act = False\n", f"be_act = False\n{S8}nsl = np.nan\n", 1),
           ("held += 1\n", f"held += 1\n{S12}if nsl == nsl:\n{S16}sl = nsl\n{S16}nsl = np.nan\n", 1),
           ("sl = ep", "nsl = ep", 2)]

MUTATIONS = {
    "be_ignored": lambda mp: mp.setattr(engine, "lvl_run", mutant(
        engine.lvl_run, [("if be > 0.0 and not be_act:", "if False:", 1)])),
    # перенос стопа вступает в силу только со следующего бара (проверка после стопа)
    "be_after_stop_check": lambda mp: mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, BE_LATE)),
    "be_ge_threshold": lambda mp: mp.setattr(engine, "lvl_run", mutant(engine.lvl_run, [
        ("(close[u] - ep) > thr", "(close[u] - ep) >= thr", 1),
        ("(ep - close[u]) > thr", "(ep - close[u]) >= thr", 1)])),
    "be_b_sigma_entry": lambda mp: mp.setattr(engine, "lvl_run", mutant(
        engine.lvl_run, [("sv2 = sigma[u]", "sv2 = sigma[e]", 1)])),
    "trend_be_ignored": lambda mp: mp.setattr(engine_trend, "trend_run", mutant(
        engine_trend.trend_run, [("tsl = max(tsl, ep)", "tsl = tsl", 1),
                                 ("tsl = min(tsl, ep)", "tsl = tsl", 1)])),
}

N = 9


def _path(b2, b3):
    """Вход на баре 1 по 100; бары 2 и 3 — (close, high, low); прочие 100 ± 0,5."""
    c, h, lo = bars(np.full(N, 100.0))
    c, h, lo = c.copy(), h.copy(), lo.copy()
    for t, (cc, hh, ll) in ((2, b2), (3, b3)):
        c[t], h[t], lo[t] = cc, hh, ll
    return c, h, lo


UP = ((101.5, 102.0, 100.8), (100.6, 101.0, 99.9))       # бар 2 включает BE, бар 3 — выход по ep


def test_be_moves_stop_to_entry_atr():
    c, h, lo = _path(*UP)
    assert run_lvl(c, h, lo, longs=[1, 4, 5], a1=2, a2=50, be=1.0, cd=2) == [(1, 3, 1), (5, N, 1)]
    assert run_lvl(c, h, lo, longs=[1, 4, 5], a1=2, a2=50, be=-1.0, cd=2) == [(1, N, 1)]


def test_be_same_bar_checked_before_stop():
    c, h, lo = _path((101.5, 102.0, 99.8), (100.0, 100.5, 99.5))
    assert run_lvl(c, h, lo, longs=[1], a1=2, a2=50, be=1.0) == [(1, 2, 1)]


def test_be_threshold_strict():
    c, h, lo = _path((101.0, 101.2, 100.8), (100.6, 101.0, 99.9))
    assert run_lvl(c, h, lo, longs=[1], a1=2, a2=50, be=1.0) == [(1, N, 1)]


def test_be_short():
    c, h, lo = _path((98.5, 99.2, 98.0), (99.4, 100.1, 99.0))
    assert run_lvl(c, h, lo, shorts=[1], a1=2, a2=50, be=1.0) == [(1, 3, -1)]


def test_be_levels_s5s6_half_take():
    c, h, lo = _path((102.5, 103.0, 101.8), UP[1])
    kw = dict(kind=1, sd=np.full(N, 3.0), tdist=np.full(N, 4.0))
    assert run_lvl(c, h, lo, longs=[1], be=0.5, **kw) == [(1, 3, 1)]      # порог 0,5·4 = 2
    c, h, lo = _path((101.9, 102.2, 101.5), UP[1])
    assert run_lvl(c, h, lo, longs=[1], be=0.5, **kw) == [(1, N, 1)]


def test_be_b_uses_current_sigma():
    c, h, lo = _path((102.0, 102.3, 101.5), UP[1])
    sg = np.full(N, 0.01)
    sg[2] = 0.03                                                          # порог 1·0,03·100 = 3
    assert run_lvl(c, h, lo, longs=[1], kind=2, a1=2, a2=50, be=1.0, sigma=sg) == [(1, N, 1)]
    sg[2] = 0.01                                                          # порог 1
    assert run_lvl(c, h, lo, longs=[1], kind=2, a1=2, a2=50, be=1.0, sigma=sg) == [(1, 3, 1)]


def test_trend_be_caps_trail_at_entry():
    c, h, lo = _path(*UP)
    assert run_trend(c, h, lo, longs=[1], isl=3.0, bth=1.0) == [(1, 3, 1)]
    assert run_trend(c, h, lo, longs=[1], isl=3.0) == [(1, N, 1)]
    cs, hs, ls = _path((98.5, 99.2, 98.0), (99.4, 100.1, 99.0))
    assert run_trend(cs, hs, ls, shorts=[1], isl=3.0, bth=1.0) == [(1, 3, -1)]


def test_trend_b_be_uses_entry_sigma():
    c, h, lo = _path(*UP)
    sg = np.full(N, 0.01)
    sg[2] = 0.03                     # по σ_вх 0,01: порог 1 → BE; по σ[2] было бы 3 — нет
    assert run_trend(c, h, lo, longs=[1], kind=2, isl=3.0, tmult=99, bth=1.0, sigma=sg,
                     base=100) == [(1, 3, 1)]
