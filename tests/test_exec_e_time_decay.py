"""V6-9 (И10, v4 engine.py:113): time_decay — после tau//2 дней в позиции тейк «net» берёт половину дистанции.
Синтетика с известным ответом. Мутации (mutations.md № 2–4 по теме исполнения): половина не включается; включается на день раньше;
не тот знак/сторона. Инвариант [low; high] (#610/#617) работает на каждом исполнении. Быстрый слой: ось → ne_znaet (эталон), не молча."""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e_run, fast_exec
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

SG = 0.01
FLAT = (100.0, 100.0, 100.0, 100.0)
UP = (100.0, 102.5, 100.0, 100.0)       # +2,5 %: между половиной (2 %) и полным тейком (4 %)
DN = (100.0, 100.0, 97.5, 100.0)
GAP = (103.0, 103.0, 100.0, 100.0)      # открытие выше половины тейка


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def go(rows, side=1, **kw):
    days = [D(r) for r in rows]
    kw = dict(a=4.0, tau=4, comm=0.0) | kw
    return run("X", days, np.ones(len(days)), np.full(len(days), SG), Params(side=side, **kw))


def ex(r):
    t = r.trades[0]
    return t.exit_day, round(t.exit_px, 6), t.reason


def test_vykl_prezhniy_put_tejk_polnyj_ne_srabatyvaet():
    r = go([FLAT, FLAT, FLAT, UP, FLAT, FLAT])
    assert ex(r)[2] == "vremya" and r.trades[0].exit_day == 4
    assert dataclasses.replace(Params(), time_decay=False) == Params()


def test_polovina_posle_poloviny_sroka_lonq():
    r = go([FLAT, FLAT, UP, UP, FLAT, FLAT], time_decay=True)      # день 2: j−ej = 2 = tau//2 — ещё полный тейк; день 3 — половина
    assert ex(r) == (3, 102.0, "take")


def test_polovina_posle_poloviny_sroka_short():
    r = go([FLAT, FLAT, DN, DN, FLAT, FLAT], side=-1, time_decay=True)
    assert ex(r) == (3, 98.0, "take")


def test_gep_cherez_polovinu_po_open_v_predelah_bara():
    r = go([FLAT, FLAT, FLAT, GAP, FLAT, FLAT], time_decay=True)
    assert ex(r) == (3, 103.0, "take")                              # гэп — по open бара, внутри [low; high]


def test_nechetnyj_tau_i_R():
    r = go([FLAT, UP, FLAT, FLAT, FLAT], time_decay=True, tau=1)    # tau//2 = 0: половина со 2-го дня
    assert ex(r) == (1, 102.0, "take")
    r = go([FLAT, FLAT, FLAT, UP, FLAT, FLAT], time_decay=True, a=None, R=2.0, b=1.0, h=1.0)    # R·b·σ√h = 2 % → половина 1 %
    assert ex(r) == (3, 101.0, "take")


def test_otkazy():
    for kw in (dict(tau=None), dict(tejk="k_srednemu")):
        with pytest.raises(ValueError):
            go([FLAT, FLAT], time_decay=True, **kw)


def test_bystryj_sloj_ne_znaet_os():
    p = Params(a=4.0, tau=4, time_decay=True)
    assert fast_exec.ne_znaet(p) == ["time_decay"]
    with pytest.raises(NotImplementedError):
        fast_exec.run_v5("X", [D(FLAT)] * 3, np.ones(3), np.full(3, SG), p)


# --- мутации: тест обязан падать ---
def test_mutatsiya_polovina_ne_vklyuchaetsya_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run.Run, "_td", lambda s, j: 1.0)
    with pytest.raises(AssertionError):
        test_polovina_posle_poloviny_sroka_lonq()


def test_mutatsiya_na_den_ranshe_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run.Run, "_td", lambda s, j: 0.5 if s.p.time_decay and j - s.ej >= s.p.tau // 2 else 1.0)
    with pytest.raises(AssertionError):
        test_polovina_posle_poloviny_sroka_lonq()


def test_mutatsiya_dvojnoj_tejk_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run.Run, "_td", lambda s, j: 2.0 if s.p.time_decay and j - s.ej > s.p.tau // 2 else 1.0)
    with pytest.raises(AssertionError):
        test_polovina_posle_poloviny_sroka_lonq()
