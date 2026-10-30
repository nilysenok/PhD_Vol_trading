"""Тестировщик PR #899 (деньги, движок): вход в T (fastbt/exec_e_t.py). Синтетика, без сети.
Мутации (check обязан падать): бары до T участвуют в стопе/тейке; цена T+10 минут; вход по open дня; 18:40 по open
(t_reshenie_v7.AUKCION) и бар входа 18:40 (exec_e_t.AUKCION). Находки: потеря входа в день разворота сигнала (умолчание автора),
точка с vhod_T не прячется от pokryto."""
import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, exec_e_run, exec_e_t, fast_etap1
from fastbt import t_reshenie_v7 as R
from fastbt.exec_e2 import Run2
from fastbt.exec_e_types import Day, Params

MODS = list(range(590, 1140, 5))
COMM = 0.0004
BT = MODS.index(720)                                          # бар 12:00


def _o(k, m):
    return 100.0 * (k + 1) + (m - 590) / 100.0


def _day(k, fn=None):
    o = np.array([_o(k, x) for x in MODS])
    d = dict(o=o, h=o + 0.5, l=o - 0.5, c=o + 0.25)
    if fn:
        fn(d)
    return Day(d["o"], d["h"], d["l"], d["c"], False, np.array(MODS))


def _cena_open(days, m=720):
    return np.array([float(d.o[MODS.index(m)]) for d in days])


def _run(sig, days, p, cena=None, **kw):
    cena = _cena_open(days) if cena is None else cena
    return exec_e.run("T", days, np.array(sig), np.full(len(days), 0.01), p, cena_T=cena, **kw)


def _tri(fn1=None):
    return [_day(0), _day(1, fn1), _day(2)]


# --- 1. выключено = v6 побайтно ---
def _sluchaynye_dni(n=40, seed=2):
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(MODS))))
        o = np.r_[c[0], c[:-1]]
        days.append(Day(o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c, False, np.array(MODS)))
    return days, rng.choice([-1, 0, 1], n, p=[0.3, 0.3, 0.4])


@pytest.mark.parametrize("kw", [dict(), dict(b=1.0, a=2.0, tau=3), dict(k=0.5, b=1.0)])
def test_vyklyuchen_raven_etalonu_pobajtno_i_RunT_ne_zovyotsya(monkeypatch, kw):
    days, sig = _sluchaynye_dni()
    sg = np.full(len(days), 0.01)
    p = Params(**kw)
    monkeypatch.setattr(exec_e_t, "RunT", None)                 # любой вызов RunT при vhod_T = None — TypeError
    a = exec_e.run("T", days, sig, sg, p)
    b = (exec_e_run.Run if not kw.get("k") else Run2)("T", days, sig, sg, p)
    for j in range(len(days)):
        b.step(j, j)
    assert a.trades == b.trades and np.array_equal(a.pos, b.pos) and a.trades


# --- 2. бары до T не задевают уровни ---
def _do_T_obval(d):
    d["l"][:BT] = 1.0
    d["h"][:BT] = 1e4


def proverka_bary_do_T(res):
    t = res.trades[0]
    assert t.entry_bar == BT and t.exit_bar > t.entry_bar and t.reason == "take" and t.exit_day == 1


def test_bary_do_T_ne_zadevayut_stop_i_tejk():
    days = _tri(_do_T_obval)
    proverka_bary_do_T(_run([0, 1, 1], days, Params(vhod_T="12:00", b=1.0, a=1.0)))


def test_mut_bary_do_T_uchastvuyut_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_t.RunT, "_scan", lambda s, i, j, d, b0: Run2._scan(s, i, j, d, 0))
    with pytest.raises(AssertionError):
        proverka_bary_do_T(_run([0, 1, 1], _tri(_do_T_obval), Params(vhod_T="12:00", b=1.0, a=1.0)))


@pytest.mark.parametrize("bar", [BT, BT + 4])
def test_stop_v_bare_vhoda_i_posle_nego_po_urovnyu(bar):
    def f(d):
        d["l"][bar] = 100.0
    r = _run([0, 1, 1], _tri(f), Params(vhod_T="12:00", b=1.0))
    t = r.trades[0]
    assert t.exit_bar == bar and t.reason == "stop" and t.exit_px == pytest.approx(_o(1, 720) * 0.99)


def test_tejk_posle_T_po_urovnyu():
    def f(d):
        d["h"][BT + 3] = 1e4
    t = _run([0, 1, 1], _tri(f), Params(vhod_T="12:00", a=1.0)).trades[0]
    assert t.exit_bar == BT + 3 and t.reason == "take" and t.exit_px == pytest.approx(_o(1, 720) * 1.01)


# --- 3. комиссия, дивиденд, конец окна ---
def test_dividend_v_den_vhoda_T_ne_platitsya_i_ne_sdvigaet_stop():
    days = _tri()
    a = _run([0, 1, 1], days, Params(vhod_T="12:00", b=1.0), div=np.array([0, 0, 0.0]))
    b = _run([0, 1, 1], days, Params(vhod_T="12:00", b=1.0), div=np.array([0, 5.0, 0]))        # отсечка в день входа, до T
    assert np.array_equal(a.gross, b.gross) and np.array_equal(a.cost, b.cost) and a.trades[0].exit_bar == b.trades[0].exit_bar


def test_dividend_perenosimoj_pozicii_platitsya_odin_raz():
    days = _tri()
    a = _run([0, 1, 1], days, Params(vhod_T="12:00"), div=np.zeros(3))
    b = _run([0, 1, 1], days, Params(vhod_T="12:00"), div=np.array([0, 0, 2.0]))
    assert b.gross[1] - a.gross[1] == pytest.approx(np.log((days[2].c[-1] + 2 * 0.87) / days[1].c[-1]) - np.log(days[2].c[-1] / days[1].c[-1]))


def test_n5_vhod_T_v_posledniy_den_okna_platit_vyhod():
    r = _run([0, 0, 1], _tri(), Params(vhod_T="12:00"))
    px, cl = _o(2, 720), _day(2).c[-1]
    assert r.cost[1] == COMM and r.cost[2] == COMM and r.gross[1] == pytest.approx(np.log(cl / px))


# --- 4. умолчание автора: вход в день выхода ---
def _markov(n, p_flip, p_flat, seed):
    rng = np.random.default_rng(seed)
    s, out = 1, []
    for _ in range(n):
        if rng.random() < p_flip:
            s = -s if s else rng.choice([-1, 1])
        if rng.random() < p_flat:
            out.append(0)
        else:
            out.append(s)
    return np.array(out)


def _zaderzhka(sig):
    """(сделок, сделок, вошедших на день позже первого дня своей стороны сигнала)."""
    days, _ = _sluchaynye_dni(len(sig), 5)
    r = _run(sig, days, Params(vhod_T="12:00", exit_signal=True))
    late = sum(1 for t in r.trades if t.entry_day > 0 and np.sign(sig[t.entry_day - 1]) == t.side)
    return len(r.trades), late


def proverka_razvorot(res, days, sig):
    """Разворот в T дня 1: обе операции по cena_T на баре T, четыре комиссии, позиции не пересекаются."""
    t0, t1 = res.trades[0], res.trades[1]
    px = days[1].o[BT]
    assert t0.reason == "razvorot" and (t0.exit_day, t0.exit_bar, t0.exit_px) == (1, BT, px)
    assert (t1.entry_day, t1.entry_bar, t1.entry_px, t1.side) == (1, BT, px, -1)
    assert [t0.fee_in, t0.fee_out, t1.fee_in] == [COMM] * 3 and t1.fee_out in (COMM, 0.0) and t0.exit_day is not None
    assert t0.exit_bar <= t1.entry_bar


def test_razvorot_v_T_obe_operacii_po_cene_T_chetyre_komissii_sdvigov_net():
    days, sig = _tri(), [1, -1, -1]
    r = _run(sig, days, Params(vhod_T="12:00"))
    proverka_razvorot(r, days, sig)
    assert r.cost.sum() == pytest.approx(4 * COMM)
    n, late = _zaderzhka(_markov(400, 0.1, 0.0, 3))
    assert n > 20 and late == 0                                 # сдвинутых на день входов нет (до правки было 88 %)


def test_mut_razvorot_bez_zakrytiya_staroj_pozicii_lovitsya(monkeypatch):
    """Мутация: на развороте старая позиция не закрывается, новая открывается поверх (две позиции сразу)."""
    days, sig = _tri(), [1, -1, -1]
    orig = exec_e_t.RunT._decide

    def dve(s, i, j, close, nb):
        if s.tr is not None and s.dir(i) == -s.tr.side:
            staraya, s.tr = s.tr, None
            orig(s, i, j, close, nb)
            if s.tr is None:
                s.tr = staraya
        else:
            orig(s, i, j, close, nb)

    monkeypatch.setattr(exec_e_t.RunT, "_scan", lambda s, i, j, d, b0: Run2._scan(s, i, j, d, b0))   # без выхода "razvorot"
    monkeypatch.setattr(exec_e_t.RunT, "_decide", dve)
    r = _run(sig, days, Params(vhod_T="12:00"))
    with pytest.raises(AssertionError):
        proverka_razvorot(r, days, sig)


def test_vyhod_stopom_do_T_ne_dolzhen_blokirovat_vhod():
    """Позиция лонг закрыта стопом на баре 0 дня 1 (раньше T), сигнал дня 1 = шорт: вход шорта в T не пересекается с выходом.
    Умолчание автора смотрит на exit_day, а не на exit_bar < бара T — вход пропущен (потеря сделки)."""
    def f(d):
        d["l"][0] = 100.0
    r = _run([1, -1, -1], [_day(0), _day(1, f), _day(2)], Params(vhod_T="12:00", b=1.0, exit_signal=True))
    assert r.trades[0].reason == "stop" and r.trades[0].exit_bar < BT
    assert len(r.trades) >= 2 and r.trades[1].entry_day == 1 and r.trades[1].side == -1


# --- 5. мутации цены/бара входа ---
def proverka_cena_T(res):
    t = res.trades[0]
    assert t.entry_bar == BT and t.entry_px == _o(1, 720)


def _skachki(d):
    d["o"][:], d["c"][:] = d["o"] + np.arange(len(MODS)) * 3.0, d["c"] + np.arange(len(MODS)) * 3.0
    d["h"][:], d["l"][:] = np.maximum(d["o"], d["c"]) + 0.1, np.minimum(d["o"], d["c"]) - 0.1


def test_cena_T_vhod_prohodit_a_mut_T_plus_10_i_open_dnya_lovyatsya():
    days = [_day(0), _day(1, _skachki), _day(2)]
    p = Params(vhod_T="12:00")
    ok = _run([0, 1, 1], days, p, cena=_cena_open(days))
    assert ok.trades[0].entry_bar == BT and ok.trades[0].entry_px == days[1].o[BT]
    with pytest.raises(ValueError, match="вне"):                 # T+10 минут: цена вне [low; high] бара T — _v_bare держит
        _run([0, 1, 1], days, p, cena=_cena_open(days, 730))
    with pytest.raises(ValueError, match="вне"):                 # вход по open дня вместо T
        _run([0, 1, 1], days, p, cena=np.array([d.o[0] for d in days]))


def test_1840_po_close_baru_1125():
    days = _tri()
    r = _run([0, 1, 1], days, Params(vhod_T="18:40"), cena=np.array([d.c[MODS.index(1125)] for d in days]))
    assert r.trades[0].entry_bar == MODS.index(1125) and r.trades[0].entry_px == _o(1, 1125) + 0.25


def test_mut_1840_po_open_cena_lovitsya(monkeypatch):
    monkeypatch.setattr(R, "AUKCION", -1)                       # 18:40 не особый: поле open
    bars = pd.DataFrame([(pd.Timestamp("2019-03-05") + pd.Timedelta(minutes=m), _o(1, m), _o(1, m) + 1, _o(1, m) - .5, _o(1, m + 5) + .25, 1)
                         for m in range(590, 1140, 10)], columns=["timestamp", "open", "high", "low", "close", "volume"])
    got = float(R.cena_T(bars, "18:40")["cena"].iloc[0])
    assert got != _o(1, 1125) + 0.25 and got == _o(1, 1120)


def test_mut_bar_vhoda_1840_pervyj_bar_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_t, "AUKCION", -1)
    d = _day(1)
    assert exec_e_t.bar_vhoda(d, 1120) == MODS.index(1120) != MODS.index(1125)       # мутация: бар входа 18:40 — до цены close(18:50)
    monkeypatch.undo()
    assert exec_e_t.bar_vhoda(d, 1120) == MODS.index(1125)


# --- 6. быстрый слой ---
def test_vhod_T_v_Params_uhodit_na_etalon_gromko():
    p = Params(vhod_T="12:00")
    assert fast_etap1._slow_only(p, None)
    with pytest.raises(ValueError, match="cena_T"):               # эталон без cena_T — отказ, не молчаливый путь ядра
        fast_etap1.run("T", _tri(), np.array([0, 1, 1]), np.full(3, 0.01), p)


def test_pokryto_tochka_s_vhod_T_ne_pokryta():
    assert fast_etap1.pokryto({}) is True
    assert fast_etap1.pokryto({"vhod_T": "12:00"}) is False
