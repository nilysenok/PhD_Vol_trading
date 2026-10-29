"""Пирамидинг V6-b: проверка по известному ответу (Т-pir, замечание проверяющего #660).
Ручные ленты 3–4 дня: размеры добавок, урезание по Σ риска ≤ 2ρ (риск = размер × расстояние от цены ноги до ТЕКУЩЕГО стопа,
Ж-111 п. 10), цены/комиссии входа и выхода ног, закрытие ног с родителем, конец окна. Числа посчитаны вручную (в комментариях)
и сверяются И с эталоном exec_e.run, И с ядром fast_exec.run_v5 — не друг с другом: одинаковая порча обоих ловится.
Общее: σ = 0,02, b = 4 → стоп 92 при входе 100, ρ = 8; comm = 0,001; сигнал 1 все дни, вход по close дня 0, reentry = False.
Мутации (одинаково в эталоне и ядре, по одной) — в описании PR: 2ρ → 3ρ, риск по стопу входа, комиссия ноги не взята.
FASTBT_NO_CACHE=1 NUMBA_CACHE_DIR=<свежий каталог> python3 -m pytest tests/test_pir_izvestnyj_otvet.py -q"""
import math

import numpy as np
import pytest

from fastbt import exec_e, fast_exec as fx
from fastbt.exec_e_types import Day, Params

C = 0.001
RUNS = [pytest.param(exec_e.run, id="etalon"), pytest.param(fx.run_v5, id="yadro")]


def day(*bars):
    o, h, l, c = (np.array(x, float) for x in zip(*bars))
    return Day(o, h, l, c, False)


def prog(run, days, p, side_sig=1.0):
    n = len(days)
    return run("T", days, np.full(n, side_sig), np.full(n, 0.02), p)


def params(**kw):
    return Params(b=4.0, comm=C, exit_signal=False, reentry=False, **kw)


def ok(x, y):
    assert x == pytest.approx(y, rel=1e-9, abs=1e-12), (x, y)


def nogi(r, k, entry, exit_, frac, fin, fout, exit_day=1, exit_bar=2):
    t = r.trades[k]
    ok(t.entry_px, entry), ok(t.exit_px, exit_), ok(t.frac, frac), ok(t.fee_in, fin), ok(t.fee_out, fout)
    assert (t.exit_day, t.exit_bar) == (exit_day, exit_bar), (k, t)


# Лента A (лонг, pir = 1,0): +1σ = 102: риск базы 8, room = 16 − 8 = 8, расстояние до стопа 102 − 92 = 10 → добавка 8/10 = 0,8.
# +2σ = 104: риск 8 + 0,8·10 = 16 = 2ρ → room 0 → второй добавки нет. Стоп 92 (гэп-вниз нет: open 103 > 92) → выход по 92.
DAYS_A = [day((100, 100, 100, 100)),
          day((100, 103, 99, 102), (102, 104.5, 101, 103), (103, 103, 91, 92))]


@pytest.mark.parametrize("run", RUNS)
def test_a_urezanie_08_i_otsutstvie_vtoroj_dobavki(run):
    r = prog(run, DAYS_A, params(pir=1.0))
    assert [t.reason for t in r.trades] == ["stop", "pir"], r.trades
    nogi(r, 0, 100, 92, 1.0, C, C * 1.0)                    # выход базы: comm·(w − Σдобавок) = comm·(1,8 − 0,8)
    nogi(r, 1, 102, 92, 0.8, C * 0.8, C * 0.8)              # добавка: цена max(open 100; 102), своя комиссия на входе и выходе
    assert (r.trades[1].entry_day, r.trades[1].entry_bar, r.trades[1].parent) == (1, 0, 0)
    ok(r.gross.sum(), math.log(102 / 100) + 1.8 * math.log(92 / 102))
    ok(r.cost.sum(), C + C * 0.8 + C * 1.8)                 # вход 0,001 + добавка 0,0008 + выход всей позиции 0,0018 = 0,0036
    ok(r.pos[1], 0.0)


# Лента A-шорт (side = −1, sig = 1): зеркально по логике, числа те же: стоп 108, +1σ вниз = 98, добавка 0,8, выход по 108.
DAYS_A_SHORT = [day((100, 100, 100, 100)),
                day((100, 101, 97.5, 98), (98, 99, 95.5, 96), (97, 109, 97, 108))]


@pytest.mark.parametrize("run", RUNS)
def test_a_short_znak_i_urezanie(run):
    r = prog(run, DAYS_A_SHORT, params(pir=1.0, side=-1))
    assert [t.reason for t in r.trades] == ["stop", "pir"], r.trades
    nogi(r, 0, 100, 108, 1.0, C, C * 1.0)
    nogi(r, 1, 98, 108, 0.8, C * 0.8, C * 0.8)
    ok(r.gross.sum(), -math.log(98 / 100) - 1.8 * math.log(108 / 98))
    ok(r.cost.sum(), C + C * 0.8 + C * 1.8)


# Лента B (лонг, pir = 0,5, окно кончается в позиции): добавка 1 на 102: room 8, 8/10 = 0,8 > 0,5 → полные 0,5.
# Добавка 2 на 104: риск 8 + 0,5·10 = 13, room 16 − 13 = 3, расстояние 104 − 92 = 12 → 3/12 = 0,25 (урезано с 0,5).
# Конец окна: w = 1,75 открыта, выход на close = 104 платит comm·1,75 на последней строке; ноги добавок остаются открытыми.
DAYS_B = [day((100, 100, 100, 100)),
          day((100, 102.5, 99, 102), (102, 104.5, 101, 104))]


@pytest.mark.parametrize("run", RUNS)
def test_b_dve_dobavki_urezanie_i_konec_okna(run):
    r = prog(run, DAYS_B, params(pir=0.5))
    assert [t.reason for t in r.trades] == [None, "pir", "pir"], r.trades
    for k, (px, a, b) in enumerate([(100, 1.0, 0), (102, 0.5, 0), (104, 0.25, 1)]):
        t = r.trades[k]
        ok(t.entry_px, px), ok(t.frac, a), ok(t.fee_in, C * a)
        assert t.exit_day is None, t
    assert (r.trades[1].entry_bar, r.trades[2].entry_bar, r.trades[2].parent) == (0, 1, 0)
    ok(r.gross.sum(), math.log(102 / 100) + 1.5 * math.log(104 / 102))   # на close 104 w = 1,75, ref = 104 → прирост 0
    ok(r.cost[0], C + C * 0.5 + C * 0.25)                   # вход и обе добавки — на строке дня 0
    ok(r.cost[1], C * 1.75)                                 # Н5: выход всей позиции на конце окна
    ok(r.pos[1], 1.75)


# Лента D (лонг, pir = 1,0, b0 = 0,5): после бара 0 (high 101,5 ≥ 101) стоп = цена входа 100 (безубыток), риск базы до ТЕКУЩЕГО стопа 0.
# Добавка на 102: room 16, расстояние 102 − 100 = 2 → min(1; 8) = 1,0 полностью. (Риск по стопу входа 92 дал бы 0,8.)
# Выход по стопу 100 (причина "bu"): база comm·(2 − 1), нога добавки comm·1.
DAYS_D = [day((100, 100, 100, 100)),
          day((100, 101.5, 99.5, 101), (101, 103, 100.5, 102), (102, 102, 99, 99.5))]


@pytest.mark.parametrize("run", RUNS)
def test_d_risk_po_tekushchemu_stopu(run):
    r = prog(run, DAYS_D, params(pir=1.0, b0=0.5))
    assert [t.reason for t in r.trades] == ["bu", "pir"], r.trades
    nogi(r, 0, 100, 100, 1.0, C, C * 1.0)
    nogi(r, 1, 102, 100, 1.0, C, C)
    ok(r.gross.sum(), math.log(102 / 100) + 2.0 * math.log(100 / 102))
    ok(r.cost.sum(), C + C + C * 2.0)


@pytest.mark.parametrize("run", RUNS)
def test_bez_pir_net_nog(run):
    r = prog(run, DAYS_A, params(pir=0.0))
    assert [t.reason for t in r.trades] == ["stop"]
    ok(r.cost.sum(), 2 * C)
