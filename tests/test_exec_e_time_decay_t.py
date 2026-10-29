"""Т-668 (тестировщик): time_decay — новая проверка Run.__init__ (tejk ≠ net) и стоп+тейк в одном баре после половины.
Мутации: снято `p.tejk != "net"` из проверки (выжила в #668); стоп и тейк в одном баре — порядок перевёрнут (mutations.md № 3)."""
import inspect
import textwrap

import numpy as np
import pytest

from fastbt import exec_e_run
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

SG = 0.01
FLAT = (100.0, 100.0, 100.0, 100.0)
BOTH = (100.0, 102.5, 97.5, 100.0)       # в одном баре: половина тейка (+2 %) и стоп (−2 %, sl=2·σ)
TEKST = "time_decay: нужен tau и tejk = net"


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def go(rows, side=1, M=None, **kw):
    days = [D(r) for r in rows]
    n = len(days)
    kw = dict(a=4.0, tau=4, comm=0.0) | kw
    return run("X", days, np.ones(n), np.full(n, SG), Params(side=side, **kw), tejk_M=M)


def M_dlya(tejk, n):
    return np.full((2, n), 104.0) if tejk == "rodnoy_uroven" else np.full(n, 104.0)


@pytest.mark.parametrize("tejk", ["rodnoy_uroven", "pol_puti", "k_srednemu"])
def test_time_decay_tejk_ne_net_otkaz_novoj_proverkoj(tejk):
    with pytest.raises(ValueError, match=TEKST):
        go([FLAT] * 6, M=M_dlya(tejk, 6), a=None, time_decay=True, tejk=tejk)


def test_time_decay_vykl_tejk_ne_net_ne_otkazyvaet():
    go([FLAT] * 6, M=M_dlya("rodnoy_uroven", 6), a=None, time_decay=False, tejk="rodnoy_uroven")


@pytest.mark.parametrize("side", [1, -1])
def test_stop_i_tejk_v_odnom_bare_posle_poloviny_stop_pobezhdaet(side):
    # день 3: j−ej > tau//2 — половина включена; бар BOTH задевает и тейк (±2 %), и стоп (∓2 %)
    r = go([FLAT, FLAT, FLAT, BOTH, FLAT, FLAT], side=side, time_decay=True, b=2.0)
    t = r.trades[0]
    assert (t.exit_day, t.reason) == (3, "stop")
    assert round(t.exit_px, 6) == (98.0 if side == 1 else 102.0)


def test_mutatsiya_uslovie_tejk_snyato_lovitsya(monkeypatch):
    src = inspect.getsource(exec_e_run.Run.__init__)
    mut = src.replace('(p.tau is None or p.tejk != "net")', "p.tau is None")
    assert mut != src
    ns = {}
    exec(textwrap.dedent(mut), exec_e_run.__dict__, ns)
    monkeypatch.setattr(exec_e_run.Run, "__init__", ns["__init__"])
    with pytest.raises(pytest.fail.Exception):      # «DID NOT RAISE» — тест (1) падает на мутации
        test_time_decay_tejk_ne_net_otkaz_novoj_proverkoj("rodnoy_uroven")
