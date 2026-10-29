"""П3b (Ж-127): параболический трейлинг (fastbt/vyhod_parabolic.py, #589) как ось автомата v2 (Params.parab, fastbt/exec_e_parab.py).
Дни из одного бара (дневной ряд), вход по close дня 0 (k = None), σ = 0,02, b = 2 → ea = 0,02·ep, isl = 2. Синтетика, без сети.
Мутации (mutations.md № 3 — стоп раньше сигнала, № 4 — гэп по уровню, заглядывание в бар): подмена правила → сценарий обязан упасть.
python3 -m pytest tests/test_exec_e_parabolic.py -q"""
import numpy as np
import pytest

from fastbt import exec_e, exec_e_parab, exec_e_run, fast_etap1
from fastbt.exec_e_types import Day, Params
from fastbt.vyhod_parabolic import NIVELI_V4, parabolic_exit

SG, B = 0.02, 2.0


def _days(o, h, l, c):
    return [Day(np.array([a], float), np.array([b], float), np.array([x], float), np.array([y], float), False)
            for a, b, x, y in zip(o, h, l, c)]


def _walk(seed, n=60):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, 0.006, n))
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.006, n)))
    return o, h, lo, c


def _auto(o, h, lo, c, sig, side, ps, pm):
    p = Params(b=B, parab=(ps, pm), stop_touch=True, side=side, exit_signal=True)
    return exec_e.run("T", _days(o, h, lo, c), sig, np.full(len(c), SG), p).trades[0]


def _ref(o, h, lo, c, sig, side, ps, pm):
    ep = c[0]
    return parabolic_exit(side, 0, len(c), ep, ep * SG, B, ps, pm, 1e10, o, h, lo, c, sig == 0)


def _same(seed, side, ps, pm, mute_sig=None):
    o, h, lo, c = _walk(seed)
    sig = np.ones(len(c))
    if mute_sig is not None:
        sig[mute_sig:] = 0
    t, (u, fill, stop) = _auto(o, h, lo, c, sig, side, ps, pm), _ref(o, h, lo, c, sig, side, ps, pm)
    if u < 0:
        return t.exit_day is None
    return t.exit_day == u and t.exit_px == pytest.approx(fill, rel=1e-12) and (t.reason == "trail") == bool(stop)


GRID = [(s, side, ps, pm, m) for s in range(25) for side in (1, -1) for ps, pm in NIVELI_V4 for m in (None, 30)]


def test_vyhod_sovpadaet_s_parabolic_exit():
    bad = [g for g in GRID if not _same(*g[:4], mute_sig=g[4])]
    assert not bad, bad[:3]
    assert sum(_ref(*_walk(s), np.ones(60), 1, 0.01, 0.1)[2] for s in range(25)) > 5      # стоп в выборке действительно срабатывает


def _scen(o, h, lo, c, sig):
    return exec_e.run("T", _days(o, h, lo, c), np.array(sig, float), np.full(len(c), SG),
                      Params(b=B, parab=(0.01, 0.10), stop_touch=True)).trades[0]


# вход 100, начальный стоп 96; день 1: уровень 96 + 0,01·(100 − 96) = 96,04; после дня 1 ext = 101, step = 0,02 → день 2: 96,1392
def test_gep_cherez_stop_po_otkrytiyu():
    t = _scen([100, 100, 90], [100, 101, 91], [100, 99, 89], [100, 100, 90], [1, 1, 1])
    assert (t.exit_day, t.reason, t.exit_px) == (2, "trail", 90.0)


def test_stop_ranshe_signala_v_odnom_bare():
    t = _scen([100, 100], [100, 100], [100, 95], [100, 97], [1, 0])
    assert (t.reason, round(t.exit_px, 4)) == ("trail", 96.04)


def test_stop_bara_bez_ego_ekstremuma():
    """Бар 1 пробивает high входа вверх и low ниже уровня 96,04: стоп считается по состоянию до бара — выход 96,04, не 96,0792."""
    t = _scen([100, 100], [100, 110], [100, 95], [100, 97], [1, 1])
    assert (t.reason, round(t.exit_px, 4)) == ("trail", 96.04)


def test_vyklyuchennaya_os_pobajtno_prezhnee():
    """parab = None: путь не медленный и совпадает с независимым быстрым ядром побайтно."""
    p = Params(b=B, a=3.0, k_tr=1.0, be=True, stop_touch=True)
    assert not exec_e._new_axes(p, None) and not fast_etap1._slow_only(p, None)
    assert fast_etap1._slow_only(Params(b=B, parab=(0.01, 0.1)), None)
    o, h, lo, c = _walk(3)
    days, sig, sgm = _days(o, h, lo, c), np.ones(len(c)), np.full(len(c), SG)
    a, f = exec_e.run("T", days, sig, sgm, p), fast_etap1.run("T", days, sig, sgm, p)
    for x in ("gross", "cost", "pos", "free", "fdr"):
        assert getattr(a, x).tobytes() == getattr(f, x).tobytes()
    assert [(t.entry_day, t.exit_day, t.exit_px, t.reason) for t in a.trades] == [(t.entry_day, t.exit_day, t.exit_px, t.reason) for t in f.trades]


@pytest.mark.parametrize("kw", [dict(parab=(0.0, 0.1)), dict(parab=(0.2, 0.1)), dict(parab=(0.01,)), dict(parab=(0.01, 0.1), b=None),
                                dict(parab=(0.01, 0.1), k_tr=1.0), dict(parab=(0.01, 0.1), be=True), dict(parab=(0.01, 0.1), chand=2.0)])
def test_nevernye_parametry(kw):
    with pytest.raises(ValueError):
        exec_e.run("T", _days(*_walk(1, 5)), np.ones(5), np.full(5, SG), Params(**{"b": B, **kw}))


# --- мутации ---
def _scen_all():
    gap = _scen([100, 100, 90], [100, 101, 91], [100, 99, 89], [100, 100, 90], [1, 1, 1])
    order = _scen([100, 100], [100, 100], [100, 95], [100, 97], [1, 0])
    return gap.exit_px == 90.0 and order.reason == "trail"


def test_scenarii_na_pravke_prokhodyat():
    assert _scen_all()


def test_mutatsiya_4_gep_po_urovnyu_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, touch: ((l <= S) if sd > 0 else (h >= S), S))
    assert not _scen_all()


def test_mutatsiya_3_stop_posle_signala_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, touch: (False, S))
    assert not _scen_all()


def test_mutatsiya_ekstremum_dnya_v_stope_togo_zhe_dnya_lovitsya(monkeypatch):
    orig = exec_e_parab.pb_level

    def peek(sd, st):                               # бар видит свой же экстремум: ext обновлён до расчёта стопа
        return orig(sd, [st[0], max(st[1], 110.0), st[2]])
    monkeypatch.setattr(exec_e_run, "pb_level", peek)
    assert not _scen_bar()


def _scen_bar():
    t = _scen([100, 100], [100, 110], [100, 95], [100, 97], [1, 1])
    return round(t.exit_px, 4) == 96.04


def _scen_vhod():
    """Лимит k = 0,5 исполнен в баре 0 дня 1 по 99 (low 96 < 97,52 = стоп после бара по ext = 100, но > начального 95,04): в баре входа стопа нет."""
    d0 = Day(np.array([100.0]), np.array([100.0]), np.array([100.0]), np.array([100.0]), False)
    d1 = Day(np.array([100.0, 100.0]), np.array([100.0, 101.0]), np.array([96.0, 99.0]), np.array([99.0, 100.0]), False)
    p = Params(b=B, k=0.5, parab=(0.5, 0.5), stop_touch=True)
    return exec_e.run("T", [d0, d1], np.ones(2), np.full(2, SG), p).trades[0]


def test_bar_vhoda_tolko_nachalnyj_stop():
    t = _scen_vhod()
    assert (t.entry_bar, t.exit_day) == (0, None)


def test_mutatsiya_stop_v_bare_vhoda_po_ext_lovitsya(monkeypatch):
    orig = exec_e_run.Run._enter

    def enter(s, *a, **kw):
        orig(s, *a, **kw)
        s.pb_new = False
    monkeypatch.setattr(exec_e_run.Run, "_enter", enter)
    t = _scen_vhod()
    assert t.exit_day == 1 and t.exit_bar == 0
