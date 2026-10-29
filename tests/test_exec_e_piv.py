"""V6-10 (И11, И12): стоп и тейк по уровням пивота S5, стоп по VWAP S6 — fastbt/exec_e_piv.py, Run (exec_e_run).
Синтетика с известным ответом: вход по close дня 0 (100), дни по одному бару. Пивот: P 99,5; S1 98; S2 96; R1 102; R2 105.
Мутации (mutations.md № 2–4, 1/23 по теме сдвига входа): уровни дня d+1; перепутанная сторона; без буфера; гэп по уровню; без нижней границы.
Быстрый слой: ось → эталон (ne_znaet), не молча."""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e_piv, exec_e_run, fast_etap1, fast_exec
from fastbt.exec_e import Params, run
from fastbt.exec_e_piv import PivUrovni
from fastbt.exec_e_types import Day

SG = 0.01
FLAT = (100.0, 100.0, 100.0, 100.0)
NAN = float("nan")


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def lv(n, **kw):
    v = dict(P=99.5, S1=98.0, S2=96.0, R1=102.0, R2=105.0) | kw
    return PivUrovni(**{k: (np.asarray(x, float) if np.ndim(x) else np.full(n, x)) for k, x in v.items()})


def go(rows, side=1, levels=None, **kw):
    days = [D(r) for r in rows]
    levels = levels if levels is not None else lv(len(days))
    return run("X", days, np.ones(len(days)), np.full(len(days), SG), Params(side=side, comm=0.0, **kw), tejk_M=levels)


def ex(r, k=0):
    t = r.trades[k]
    return t.exit_day, None if t.exit_px is None else round(t.exit_px, 6), t.reason


def bar(lo=100.0, hi=100.0, o=100.0, c=100.0):
    return (o, hi, lo, c)


# --- выкл. = v5 побайтно ---
def test_vykl_prezhniy_put_pobajtno():
    rng = np.random.default_rng(7)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 40))
    rows = [(x, x * 1.01, x * 0.99, x) for x in c]
    days, sig, sigma = [D(r) for r in rows], np.sign(rng.normal(size=40)), np.full(40, SG)
    p = Params(a=2.0, b=1.0, comm=0.0004)
    a = run("X", days, sig, sigma, p)
    b = run("X", days, sig, sigma, p, tejk_M=lv(40))        # ось выключена: ряд уровней не читается
    assert a.trades == b.trades and a.n_bez_tejka == b.n_bez_tejka
    for x in ("gross", "cost", "pos", "free"):
        assert np.array_equal(getattr(a, x), getattr(b, x))
    assert (Params.piv_sl, Params.piv_tp, Params.piv_buf, Params.vwap_m) == (None, None, 0.0, None)


def test_stop_next_lonq():
    assert ex(go([FLAT, bar(97.9)], piv_sl="next")) == (1, 98.0, "stop")


def test_stop_half_lonq():
    assert ex(go([FLAT, bar(99.1), bar(98.9)], piv_sl="half")) == (2, 99.0, "stop")


def test_stop_bufer_lonq():
    r = go([FLAT, bar(97.5), bar(96.9)], piv_sl="next", piv_buf=0.5)      # расстояние 2·1,5 = 3 → стоп 97
    assert ex(r) == (2, 97.0, "stop")


@pytest.mark.parametrize("tp,T", [("P", 101.5), ("R", 104.0), ("half", 102.0)])
def test_tejk_lonq(tp, T):
    assert ex(go([FLAT, bar(hi=T + 0.1)], piv_tp=tp)) == (1, T, "take")


def test_stop_ranshe_tejka_v_odnom_bare():
    assert ex(go([FLAT, bar(97.9, 104.1)], piv_sl="next", piv_tp="R")) == (1, 98.0, "stop")


def test_stop_next_short():
    assert ex(go([FLAT, bar(hi=103.1)], side=-1, piv_sl="next")) == (1, 103.0, "stop")      # |R2 − R1| = 3, не |S1 − S2| = 2


def test_stop_half_short():
    assert ex(go([FLAT, bar(hi=101.6)], side=-1, piv_sl="half")) == (1, 101.5, "stop")


@pytest.mark.parametrize("tp,T", [("P", 97.5), ("R", 96.0), ("half", 98.0)])
def test_tejk_short(tp, T):
    assert ex(go([FLAT, bar(lo=T - 0.1)], side=-1, piv_tp=tp)) == (1, T, "take")


def test_vwap_stop_lonq():
    r = go([FLAT, bar(98.9)], piv_sl="vwap", vwap_m=5.0, levels=lv(2, vwap=104.0, dev=1.0))     # полоса 99 → расстояние 1
    assert ex(r) == (1, 99.0, "stop")


def test_vwap_stop_short():
    r = go([FLAT, bar(hi=101.1)], side=-1, piv_sl="vwap", vwap_m=5.0, levels=lv(2, vwap=96.0, dev=1.0))
    assert ex(r) == (1, 101.0, "stop")


def test_vwap_polosa_vyshe_ceny_kak_v4():
    r = go([FLAT, bar(97.9)], piv_sl="vwap", vwap_m=3.0, levels=lv(2, vwap=104.0, dev=1.0))    # полоса 101: |100 − 101| = 1 → 99
    assert ex(r) == (1, 99.0, "stop")


def test_nechislovye_urovni_nizhnyaya_granitsa():
    nan = lv(2, P=NAN, S1=NAN, S2=NAN, R1=NAN, R2=NAN)
    assert ex(go([FLAT, bar(99.85)], piv_sl="next", levels=nan)) == (1, 99.9, "stop")            # 0,001·close
    assert ex(go([FLAT, bar(hi=100.15)], piv_tp="R", levels=nan)) == (1, 100.1, "take")


def test_s2_nan_zapasnoj_uroven():
    r = go([FLAT, bar(97.9)], piv_sl="next", levels=lv(2, S2=NAN))      # S2 = S1 − |P − S1| = 96,5 → расстояние 1,5 → стоп 98,5
    assert ex(r) == (1, 98.5, "stop")


# --- данные ≤ close d сигнала, гэп, инвариант ---
def test_urovni_dnya_signala_ne_d_plus_1():
    lev = lv(3, S1=[98.0, 90.0, 90.0], S2=[96.0, 80.0, 80.0])
    assert ex(go([FLAT, bar(97.9), FLAT], piv_sl="next", levels=lev)) == (1, 98.0, "stop")


def test_limitnyj_vhod_urovni_dnya_signala():
    lev = lv(3, S1=[98.0, 90.0, 90.0], S2=[96.0, 80.0, 80.0])
    r = go([FLAT, bar(98.5, 100.0, c=98.6), bar(96.9, 99.0, o=99.0)], levels=lev, piv_sl="next", k=1.0, f="propusk")
    assert r.trades[0].entry_px == 99.0 and ex(r) == (2, 97.0, "stop")


def test_gep_cherez_stop_po_open():
    assert ex(go([FLAT, (97.0, 97.5, 96.5, 97.0)], piv_sl="next")) == (1, 97.0, "stop")


def test_gep_cherez_tejk_po_open():
    assert ex(go([FLAT, (103.0, 103.5, 102.8, 103.0)], piv_tp="P")) == (1, 103.0, "take")


def test_otkazy():
    for kw in (dict(piv_sl="next", b=1.0), dict(piv_sl="x"), dict(piv_sl="vwap"), dict(piv_sl="next", vwap_m=2.0),
               dict(piv_tp="P", a=2.0), dict(piv_tp="y"), dict(piv_sl="next", be=True)):
        with pytest.raises(ValueError):
            go([FLAT, FLAT], **kw)
    with pytest.raises(ValueError):
        go([FLAT, FLAT], piv_sl="next", levels=lv(5))                       # ряд не той длины
    with pytest.raises(ValueError):
        run("X", [D(FLAT)] * 2, np.ones(2), np.full(2, SG), Params(piv_sl="next"))     # нет PivUrovni


def test_bystryj_sloj_ne_znaet_os_i_otdayot_etalonu():
    p = Params(piv_sl="next", piv_tp="P", comm=0.0)
    assert fast_exec.ne_znaet(p) == ["piv_sl", "piv_tp"]
    days, args = [D(FLAT), D(bar(97.9))], (np.ones(2), np.full(2, SG), p)
    with pytest.raises(NotImplementedError):
        fast_exec.run_v5("X", days, *args)
    a, b = fast_etap1.run("X", days, *args, tejk_M=lv(2)), run("X", days, *args, tejk_M=lv(2))
    assert a.trades == b.trades and ex(a) == (1, 98.0, "stop")
    assert dataclasses.replace(p, piv_sl=None, piv_tp=None) == Params(comm=0.0)


# --- мутации: тест обязан падать ---
def _wrap(monkeypatch, f):
    real = exec_e_piv.enter_levels
    monkeypatch.setattr(exec_e_run, "enter_levels", lambda p, lev, sd, px, d, c, t: f(real, p, lev, sd, px, d, c, t))


def test_mutatsiya_urovni_d_plus_1_lovitsya(monkeypatch):
    _wrap(monkeypatch, lambda real, p, lev, sd, px, d, c, t: real(p, lev, sd, px, d + 1, c, t))
    with pytest.raises(AssertionError):
        test_urovni_dnya_signala_ne_d_plus_1()


def test_mutatsiya_pereputana_storona_lovitsya(monkeypatch):
    def f(real, p, lev, sd, px, d, c, t):                   # расстояние лонга (|S1 − S2|) применено к шорту
        S, T = real(p, lev, 1, px, d, c, t)
        return px - sd * (px - S), T
    _wrap(monkeypatch, f)
    with pytest.raises(AssertionError):
        test_stop_next_short()


def test_mutatsiya_bez_bufera_lovitsya(monkeypatch):
    _wrap(monkeypatch, lambda real, p, lev, sd, px, d, c, t: real(dataclasses.replace(p, piv_buf=0.0), lev, sd, px, d, c, t))
    with pytest.raises(AssertionError):
        test_stop_bufer_lonq()


def test_mutatsiya_gep_po_urovnyu_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, touch: ((l < S) if sd > 0 else (h > S), S))
    with pytest.raises((AssertionError, ValueError)):       # цена 98 вне [96,5; 97,5]: инвариант исполнения
        test_gep_cherez_stop_po_open()


def test_mutatsiya_bez_nizhnej_granicy_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_piv, "MIN_D", 0.0)
    with pytest.raises(AssertionError):
        test_nechislovye_urovni_nizhnyaya_granitsa()
