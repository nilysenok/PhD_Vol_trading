"""V6-11c (Ж-111 п. 3, И16): reentry_den на стоп-заявке тренда — повторный вход в тот же день после внутридневного выхода.
Правило: пара вечера d−1 (buy-stop 102 / sell-stop 98) стоит на барах строго после бара выхода; не более одного повторного входа за день;
гэп через уровень — по open (№ 4); уровень вечера d (известный только после close) не используется. T > 1 и smena на стоп-заявке — свёртка в канон.
Мутации: пара дня d вместо d−1; ось принята ядром быстрого слоя; лимитный reenter Run вызван на RunStop.
python3 -m pytest tests/test_exec_e_stop_reentry.py -q"""
import dataclasses

import numpy as np
import pytest
from test_exec_e_stop_osi import N0, NAN, P, _day, _flat, _rnd

from fastbt import exec_e, exec_e_stop, fast_etap1, fast_exec

BARS = [(101, 102.5, 101, 102.2),       # 0: buy-stop 102 → вход по 102 (low 101 выше стопа 100,98)
        (102.2, 102.3, 100.5, 100.6),   # 1: стоп 100,98 → выход; пара активна только с бара 2
        (100.6, 102.4, 101.2, 102.2),   # 2:buy-stop 102 → повторный вход по 102
        (102.2, 102.3, 100.0, 100.5),   # 3: стоп 100,98 → выход
        (100.5, 103.0, 100.4, 102.5)]   # 4: пара снова коснулась, но второго повторного входа за день нет


def _go(bars, p, up15=NAN, f=exec_e.run):
    days = [_flat() for _ in range(N0)] + [_day(*zip(*bars))]
    n = len(days)
    u = {k: np.full(n, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][N0 - 1], u["dn"][N0 - 1], u["up"][N0] = 102.0, 98.0, up15
    return f("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u), days


def test_povtornyj_vkhod_posle_vnutridnevnogo_vykhoda():
    r, _ = _go(BARS, P(b=1.0, reentry_den=True))
    a, b = r.trades
    assert (a.entry_bar, a.exit_bar, a.reason, round(a.exit_px, 9)) == (0, 1, "stop", 100.98)
    assert (b.entry_bar, b.entry_px, b.exit_bar, b.reason, b.d_sig) == (2, 102.0, 3, "stop", N0 - 1)
    assert len(r.trades) == 2 and (r.n_orders, r.n_fills) == (2, 2)          # вечерняя заявка + повторная (последний день вечерней нет); одна на день
    assert len(_go(BARS, P(b=1.0))[0].trades) == 1                           # без оси — как v5


def test_gep_cherez_uroven_po_open():
    bars = [BARS[0], BARS[1], (103, 104, 102.6, 103.5)]                      # open 103 выше buy-stop 102 → вход по open
    assert _go(bars, P(b=1.0, reentry_den=True))[0].trades[1].entry_px == 103.0


def test_bez_povtora_posle_vykhoda_na_poslednem_bare():
    r, _ = _go([BARS[0], (102.2, 102.3, 100.5, 100.6)], P(b=1.0, reentry_den=True))
    assert len(r.trades) == 1                                                # после бара выхода баров нет


def test_uroven_dnya_d_ne_ispolzuetsya_mutaciya(monkeypatch):
    p = P(b=1.0, reentry_den=True)
    assert len(_go(BARS, p, up15=120.0)[0].trades) == 2                      # уровень дня d (120) ни на что не влияет
    orig = exec_e_stop.RunStop._pair
    monkeypatch.setattr(exec_e_stop.RunStop, "_pair", lambda s, i: orig(s, i + 1))
    assert len(_go(BARS, p, up15=120.0)[0].trades) != 2                      # пара дня d: повторного входа нет → тест её ловит


def test_tsena_v_low_high_s_osyami():
    p = P(b=2.0, a=3.0, b0=1.0, pir=0.5, ed_s="atr", chand=1.5, k_e=0.1, reentry_den=True, comm=0.0004)
    n = k = 0
    for seed in range(8):
        B, u = _rnd(seed)
        r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), p, urovni=u)
        k += len({(t.entry_day) for t in r.trades}) != len(r.trades)        # были дни с двумя сделками
        for t in r.trades:
            for dy, br, px in ((t.entry_day, t.entry_bar, t.entry_px), (t.exit_day, t.exit_bar, t.exit_px)):
                if dy is not None and px != B[dy].c[-1]:
                    assert B[dy].l[br] - 1e-9 <= px <= B[dy].h[br] + 1e-9
                    n += 1
        by_day = {}
        for t in r.trades:
            if t.reason != "pir":
                by_day[t.entry_day] = by_day.get(t.entry_day, 0) + 1
    assert n > 20 and k > 0 and max(by_day.values()) <= 2        # не больше двух входов в день: вечерняя заявка и один повтор


def test_T_bolshe_odnogo_i_limitnyj_reentry_ne_dlya_stop_zayavki():
    for bad in (dict(T=3), dict(T=2, reentry_den=True)):
        with pytest.raises(ValueError):
            _go(BARS, dataclasses.replace(P(b=1.0), **bad))
    with pytest.raises(ValueError):
        exec_e.run("T", [_flat()] * 3, np.zeros(3), np.full(3, 0.01), exec_e.Params(reentry_den=True))     # лимитный путь: k не задан


def test_limitnyj_place_na_runstop_zapreshchen():
    with pytest.raises(RuntimeError):
        exec_e_stop.RunStop._place(None)


def test_reentry_na_stop_zayavke_idet_etalonom(monkeypatch):
    p = P(b=1.0, reentry_den=True)
    assert fast_exec.ne_znaet(p) == ["reentry_den"] and not fast_etap1._yadro_v5(p, None, None, None)
    a, b = _go(BARS, p)[0], _go(BARS, p, f=fast_etap1.run)[0]
    assert len(a.trades) == 2 and a.trades == b.trades and a.cost.tobytes() == b.cost.tobytes()
    monkeypatch.setattr(exec_e_stop, "stop_novye", lambda p: [])
    assert fast_exec.ne_znaet(p) == []                                         # подложенная ошибка: ядро берёт ось → тест выше упал бы
