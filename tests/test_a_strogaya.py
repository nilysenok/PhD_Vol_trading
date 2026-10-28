"""A_strogaya (Ж-086 п. 1–2, Ж-088 п. 4): вход rynok по open 10:00 d+1, выход по open 10:00 x+1, флаг. Синтетика, без данных.
check_* — на правильном коде; test_mut_* — подложенная ошибка (check обязан упасть). Мутации: rynok по close d; выход по close x;
бар 09:55 вместо 10:00; якорь исполняется строго; rynok справочный; точка подключения без vhod_open.
Якорь не менялся: tests/test_etap1_run_yakor.py и tests/test_yakor_v1.py остаются зелёными (запуск — в описании PR).
Портфель A, последний день окна и данные train4 — tests/test_a_strogaya_ryady.py."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, exec_e_vremya
from fastbt import progon_osnovnaya as P
from fastbt.etap1_eval import YAKOR
from fastbt.etap1_eval_v2 import NOVYE, EvaluatorV2
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

LN = np.log


def dayx(base, n=8, t0=590, step=0.1, **bar):
    """Бары от 09:50 (t0) по 5 минут; open бара b = base + step·b (по open видно, какой бар взят); bar={b: (o, h, l, c)}."""
    o = base + step * np.arange(n)
    c, h, l = o + 0.05, o + 0.06, o - 0.01
    d = Day(o.copy(), h, l, c, False, None if t0 is None else t0 + 5 * np.arange(n))
    for b, v in bar.items():
        d.o[int(b[1:])], d.h[int(b[1:])], d.l[int(b[1:])], d.c[int(b[1:])] = v
    return d


def rn(days, sig, **kw):
    kw.setdefault("comm", 0.0)
    return run("T", days, np.array(sig), np.full(len(days), 0.02), Params(**kw))


def lenta():
    return [dayx(100.0), dayx(101.0), dayx(103.0), dayx(104.0)]       # open бара 10:00 (b = 2): 100.2 / 101.2 / 103.2 / 104.2


def check_vhod_i_vyhod_open():
    r = rn(lenta(), [1, 0, 0, 0], vhod_open=True, t_vyhod="open2")
    tr = r.trades[0]
    assert (tr.entry_day, tr.entry_bar, tr.entry_px) == (1, 2, pytest.approx(101.2))      # open 10:00 d+1, а не close d (100.35)
    assert (tr.exit_day, tr.exit_bar, tr.exit_px, tr.reason) == (2, 2, pytest.approx(103.2), "signal")    # open 10:00 x+1
    assert r.gross.sum() == pytest.approx(LN(103.2 / 101.2))                      # цены сделки, а не close


def test_vhod_i_vyhod_open():
    check_vhod_i_vyhod_open()


def test_mut_rynok_po_close_d(monkeypatch):
    orig = exec_e_run.Run._decide

    def decide(s, i, j, close, nb):                                  # мутация: vhod_open игнорируется, вход по close d
        p, s.p = s.p, replace(s.p, vhod_open=False)
        try:
            orig(s, i, j, close, nb)
        finally:
            s.p = p
    monkeypatch.setattr(exec_e_run.Run, "_decide", decide)
    with pytest.raises(AssertionError):
        check_vhod_i_vyhod_open()


def test_mut_vyhod_po_close_x(monkeypatch):
    monkeypatch.setattr(exec_e_vremya, "vyhod_plan", lambda tv, pr: (0, None))      # мутация: выход по close x
    with pytest.raises(AssertionError):
        check_vhod_i_vyhod_open()


def test_mut_bar_0955_vmesto_1000(monkeypatch):
    monkeypatch.setitem(exec_e_vremya.HHMM, "open2", 595)             # мутация: бар 09:55
    with pytest.raises(AssertionError):
        check_vhod_i_vyhod_open()


def check_yakor_po_close():
    tr = rn(lenta(), [1, 0, 0, 0]).trades[0]                          # умолчания: вход по close d (якорь не менялся)
    assert (tr.entry_day, tr.entry_px) == (0, pytest.approx(100.75))


def test_yakor_po_close_d():
    check_yakor_po_close()


def test_mut_yakor_ispolnyaetsya_strogo(monkeypatch):
    orig = exec_e_run.Run._decide

    def decide(s, i, j, close, nb):
        p, s.p = s.p, replace(s.p, vhod_open=True)                    # мутация: открытый вход включён без флага
        try:
            orig(s, i, j, close, nb)
        finally:
            s.p = p
    monkeypatch.setattr(exec_e_run.Run, "_decide", decide)
    monkeypatch.setattr(exec_e, "_new_axes", lambda p, ed: True)     # и движок Run2, который умеет открытый вход
    with pytest.raises(AssertionError):
        check_yakor_po_close()


def test_shorty_znak():
    r = rn(lenta(), [-1, 0, 0, 0], vhod_open=True, t_vyhod="open2")
    assert r.trades[0].side == -1
    assert r.gross.sum() == pytest.approx(-LN(103.2 / 101.2))        # цена выросла — шорт в минусе


def test_stop_v_bare_vhoda():
    days = lenta()
    days[1] = dayx(101.0, b2=(101.2, 101.3, 98.0, 99.0))             # тот же бар, что вход: low ниже стопа 101.2·0.98
    tr = rn(days, [1, 0, 0, 0], vhod_open=True, b=1.0).trades[0]
    assert (tr.reason, tr.exit_day, tr.exit_bar) == ("stop", 1, 2) and tr.exit_px == pytest.approx(101.2 * 0.98)


def test_komissiya_vhod_i_vyhod():
    r = rn(lenta(), [1, 0, 0, 0], vhod_open=True, t_vyhod="open2", comm=0.0004)
    assert r.cost.sum() == pytest.approx(0.0008)


@pytest.mark.parametrize("t,b,den", [([590, 595, 605, 610, 615, 620, 625, 630], 2, 1), (None, 0, 1), ([590, 595], None, 2)],
                         ids=["net_bara_1000", "net_vremeni", "net_barov_posle_1000"])
def test_zapasnoe_pravilo_vhoda(t, b, den):
    days = lenta()
    days[1] = replace(dayx(101.0, n=len(t) if t else 8, t0=None), t=None if t is None else np.array(t))
    tr = rn(days, [1, 0, 0, 0], vhod_open=True, t_vyhod="open2").trades[0]
    assert tr.entry_day == den and tr.entry_bar == (b if b is not None else 2)     # нет баров после 10:00 -> open 10:00 следующего дня


def test_check_params():
    with pytest.raises(ValueError, match="vhod_open"):
        rn(lenta(), [1, 0, 0, 0], vhod_open=True, k=0.5)             # лимитный вход k_e — без изменений, vhod_open к нему не применим
    with pytest.raises(ValueError, match="vhod_open"):
        rn(lenta(), [1, 0, 0, 0], vhod_open=True, check="zakrytie")


# --- флаг основной строки и точка подключения EvaluatorV2 ---
def test_flag_rynok_osnovnoj():
    assert P.RYNOK_ISPOLNENIE == "open_d1"
    assert P.osnovnaya({"t_vyhod": "open2", "vhod": "rynok"}) is True
    assert P.osnovnaya({"t_vyhod": "close", "vhod": "rynok"}) is False      # close x справочный при любом входе


def test_mut_flag_rynok_spravochnyj(monkeypatch):
    monkeypatch.setattr(P, "RYNOK_ISPOLNENIE", "close_d")
    with pytest.raises(AssertionError):
        test_flag_rynok_osnovnoj()


def _vhod_open(vhod, tv):
    ev = EvaluatorV2.__new__(EvaluatorV2)
    ev.inp, ev.comm = SimpleNamespace(tick={"T": 0.01}), 0.0004
    pt = {**YAKOR, "k_s": 1.5, "proverka": "bars5m", "vhod": vhod}
    return ev._params(pt, "T", 5.0, False, {**NOVYE, "t_vyhod": tv}, {}).vhod_open


def check_tochka_podklyucheniya():
    assert _vhod_open("rynok", "open2") is True and _vhod_open("rynok", "1500") is True
    assert _vhod_open("rynok", "close") is False and _vhod_open("0.5", "open2") is False


def _row(kod, vhod="rynok"):
    return {"klyuch": {"pt": {"t_vyhod": "open2", "vhod": vhod}}, "kod": kod, "istochnik": "sobol1"}


def test_rynok_po_kodu_stroki():
    assert P.osnovnaya_stroki(_row("d4711aa5" + "0" * 32)) is False       # проходы 1–2: rynok по close d
    assert P.osnovnaya_stroki(_row("f" * 40)) is True and P.osnovnaya_stroki(_row("d4711aa5", "0.5")) is True
    with pytest.raises(ValueError, match="kod"):
        P.osnovnaya_stroki({"klyuch": {"pt": {"t_vyhod": "open2", "vhod": "rynok"}}})


def test_tochka_podklyucheniya():
    check_tochka_podklyucheniya()


def test_mut_tochka_bez_vhod_open(monkeypatch):
    orig = EvaluatorV2._params
    monkeypatch.setattr(EvaluatorV2, "_params", lambda s, *a: replace(orig(s, *a), vhod_open=False))
    with pytest.raises(AssertionError):
        check_tochka_podklyucheniya()


