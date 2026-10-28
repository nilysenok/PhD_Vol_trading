"""CE-6: E, длинная сторона, строгое исполнение (fastbt/sem_e_strogiy.py). Синтетика с известным ответом, сети нет."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sem_e, sem_e_strogiy as ses
from fastbt.exec_e import Day, Params

T = np.arange(590, 590 + 5 * 12, 5)           # 09:50 …; бар 10:00 — индекс 2


def _day(base):
    o = base + 0.01 * np.arange(12)
    return Day(o, o + 0.05, o - 0.05, o + 0.005, False, t=T)


def _inp(sigma=0.5):
    n = 9
    dates = pd.bdate_range("2018-01-09", periods=n)
    days = [_day(100.0 + 10 * k) for k in range(n)]
    pos = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0.0])
    p = sem_e.Pair("AAA", "S1", 0, dates, pos, np.zeros(n), None, days)
    inp = sem_e.Inputs([p], {"s": {"AAA": np.full(n, sigma)}}, pd.Series(0.0, index=dates))
    return p, inp, dates[-1]


OBJ = {"sigma": "s"}


def _trade(par, comm=0.0):
    p, inp, end = _inp()
    r, *_ = ses.run_dlinnaya(p, inp, OBJ, par, comm, end)
    assert len(r.trades) == 1
    return p.days, r.trades[0]


def test_vhod_open_1000_d_plus_1_i_vyhod_open_x_plus_1():
    days, tr = _trade(dict(q=0.0))
    assert (tr.entry_day, tr.exit_day, tr.reason) == (2, 5, "signal")   # сигнал на close 1 → вход день 2; sig=0 на дне 4 → выход день 5
    assert tr.entry_px == pytest.approx(ses.cena_vhoda(days, 1, {})) == pytest.approx(days[2].o[2])
    assert tr.exit_px == pytest.approx(ses.cena_vyhoda(days, 4, {}, "signal")) == pytest.approx(days[5].o[2])
    assert tr.entry_px != pytest.approx(days[1].c[-1]) and tr.exit_px != pytest.approx(days[4].c[-1])


def test_tau_vyhod_open_x_plus_1():
    days, tr = _trade(dict(q=0.0, tau=1))
    assert tr.reason == "vremya" and tr.exit_day == tr.entry_day + 2
    assert tr.exit_px == pytest.approx(ses.cena_vyhoda(days, tr.entry_day + 1, {}, "tau"))


def test_mutatsiya_close_vmesto_open_lovitsya(monkeypatch):
    """М-CE3/М-CE4: движок со входом по close d и выходом по close x даёт другие цены — тест выше упал бы."""
    monkeypatch.setattr(ses, "params_strogie", lambda par, comm, sp=False: Params(comm=comm / 100))
    days, tr = _trade(dict(q=0.0))
    assert tr.entry_px == pytest.approx(days[1].c[-1]) and tr.entry_px != pytest.approx(days[2].o[2])
    assert tr.exit_px != pytest.approx(days[5].o[2])


def test_f_zakrytie_d1_tolko_spravka():
    p, inp, end = _inp()
    with pytest.raises(ValueError, match="справочная"):
        ses.run_dlinnaya(p, inp, OBJ, dict(k=0.5, f="zakrytie_d1"), 0.0, end)
    ses.run_dlinnaya(p, inp, OBJ, dict(k=0.5, f="zakrytie_d1"), 0.0, end, spravka=True)


def test_e1_limit_po_baram_d_plus_1():
    p, inp, end = _inp(sigma=0.0005)           # L = P_d·(1 − 0,5·σ̂) чуть ниже: бары d+1 (на +10) выше → вход не по close d
    pr = ses.params_strogie(dict(k=0.5, f="propusk"), 4.0)
    assert pr.k == 0.5 and not pr.vhod_open and pr.t_vyhod == "open2"
    with pytest.raises(ValueError):
        ses.cena_vhoda(p.days, 1, dict(family="E1", k=0.5))


def test_comissiya_i_znak():
    _, t0 = _trade(dict(q=0.0), comm=0.0)
    p, inp, end = _inp()
    ex0, _, c0 = ses.pair_e(p, inp, OBJ, dict(q=0.0), 0.0, end)
    ex1, _, c1 = ses.pair_e(p, inp, OBJ, dict(q=0.0), 0.04, end)
    assert c0 == c1 == 2.0                      # вход и выход
    assert ex0.sum() - ex1.sum() == pytest.approx(2 * 0.0004)
    assert (ses.pair_e(p, inp, OBJ, dict(q=0.0), 0.0, end)[1] >= 0).all()   # длинная сторона: позиция неотрицательна


def test_net_dnya_vne_okna_i_prichina():
    p, _, _ = _inp()
    with pytest.raises(ValueError):
        ses.cena_vhoda(p.days, len(p.days) - 1, {})      # d+1 за окном — цены нет, а не nan
    with pytest.raises(ValueError):
        ses.cena_vyhoda(p.days, 1, {}, "stop")


def test_pustaya_lenta_signal_nol_net_sdelok():
    p, inp, end = _inp()
    p.pos[:] = 0.0
    r, *_ = ses.run_dlinnaya(p, inp, OBJ, dict(q=0.0), 0.04, end)
    assert r.trades == [] and not r.cost.any()
