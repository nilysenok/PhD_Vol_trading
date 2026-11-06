"""LGBM волна D: окна, распад, блоки обновления — известные ответы, главный тест обрезки на реальном мире Б-2 + F4, мутации (MD1–MD3, MD8, MD9)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import select
from models_v2.s6.lgbm import c_hp, c_l0, d_fit, d_win, job, world
from models_v2.s6.lgbm.d_win import Tm

HP = c_hp.HpC(8, 10, 2.0, 0.05, 0.7, 0.8, 3, 120, 42)


@pytest.fixture(scope="module")
def W():
    return world.load()


def spec(W):
    return c_l0.spec_of(W, "t_d", "F4")


def row_of(W, ds):
    return int(np.flatnonzero(W["F"].dates == np.datetime64(ds))[0])


def damaged(sp, y_from, x_from):
    X, y = sp.X.copy(), sp.y.copy()
    X[x_from:] = 1e6
    y[y_from:] = 1e6
    return dataclasses.replace(sp, X=X, y=y)


def test_window_is_last_w_pairs_and_mutation_caught():
    idx = np.arange(100)
    assert (d_win.nearest(idx, 50.0, 10) == np.arange(45, 55)).all()          # известный ответ: ближайшие к центру, ничья — раньше
    assert (d_win.nearest(idx[:60], 59.0, 10) == np.arange(50, 60)).all()       # центр справа от всех пар (прогноз): последние W
    assert not (d_win.nearest(idx, 50.0, 10) == np.arange(90, 100)).all()       # мутация «последние W» при центре внутри — другой набор
    assert (d_win.nearest(idx, 50.0, 0) == idx).all() and (d_win.nearest(idx, 50.0, 500) == idx).all()


def test_decay_known_answer_and_mutation_caught():
    w = d_win.weights_full(11, 5, 2.0)
    assert w[5] == 1.0 and w[3] == 0.5 and w[7] == 0.5 and w[1] == 0.25
    assert w[3] < w[4] < w[5] > w[6] > w[7]                                      # свежее тяжелее
    assert d_win.weights_full(5, 2, 0) is None
    mut = 0.5 ** (-np.abs(np.arange(11) - 5) / 2.0)                             # мутация: старое тяжелее
    assert not mut[3] < mut[4]


def test_blocks_anchor_first_and_mutation_caught():
    dates = np.array(pd.bdate_range("2021-01-25", periods=12), dtype="datetime64[ns]")   # пн 25.01 … вт 09.02
    grid = np.arange(100, 112)
    wk = d_win.blocks(grid, d_win.block_ids(dates, "w"))
    assert [len(b[1]) for b in wk] == [5, 5, 2] and [a for a, _ in wk] == [100, 105, 110]
    mo = d_win.blocks(grid, d_win.block_ids(dates, "m"))
    assert [len(b[1]) for b in mo] == [5, 7] and mo[1][0] == 105                  # якорь второго месяца — 01.02
    ad = d_win.blocks(grid, d_win.block_ids(dates, "adapt", np.r_[np.zeros(2, bool), True, np.zeros(9, bool)]))
    assert [a for a, _ in ad] == [100, 102, 105]                                  # смена режима открывает блок
    assert all(a == min(ds) and all(d >= a for d in ds) for a, ds in wk + mo + ad)
    assert [a for a, _ in d_win.blocks(grid, d_win.block_ids(dates, "d"))] == list(grid)
    assert max(wk[0][1]) > 100 and max(wk[0][1]) != wk[0][0]                      # мутация «якорь = последняя дата блока» дала бы a > min(ds)


def test_unbounded_window_equals_base_bitwise(W):
    F, sp = W["F"], spec(W)
    Y, y = 2020, 2018
    base = job.fold_q(F, sp, W["eps"], HP, Y, y)
    assert d_fit.fold_q_d(F, sp, W["eps"], HP, Tm(), Y, y)[1].mean() == base
    assert d_fit.fold_q_d(F, sp, W["eps"], HP, Tm(win=10 ** 6), Y, y)[1].mean() == base
    assert d_fit.fold_q_d(F, sp, W["eps"], HP, Tm(win=300), Y, y)[1].mean() != base   # мутация: ограниченное окно — не база
    d = row_of(W, "2022-06-01")
    a, b = job.day_fc(F, sp, HP, d), d_fit.block_fc(F, sp, HP, Tm(), d, [d])[d]
    assert a["f"] == b["f"] and a["s2"] == b["s2"] and a["k"] == b["k"]


def test_day_window_invariant_to_rows_after_d(W):
    F, sp = W["F"], spec(W)
    for ds in ("2018-06-01", "2022-06-01", "2025-06-02"):
        d = row_of(W, ds)
        for tm in (Tm(win=504), Tm(hl=756.0), Tm(win=252, hl=756.0)):
            a = d_fit.block_fc(F, sp, HP, tm, d, [d])[d]
            b = d_fit.block_fc(F, damaged(sp, d, d + 1), HP, tm, d, [d])[d]
            assert a["f"] == b["f"] and a["s2"] == b["s2"] and a["k"] == b["k"]


def test_block_forecast_invariant_to_rows_after_anchor(W):
    F, sp = W["F"], spec(W)
    a = row_of(W, "2022-06-06")
    ds = [a, a + 1, a + 2, a + 3, a + 4]
    for tm in (Tm(), Tm(win=504)):
        r1 = d_fit.block_fc(F, sp, HP, tm, a, ds)
        r2 = d_fit.block_fc(F, damaged(sp, a, ds[-1] + 1), HP, tm, a, ds)         # цели с якоря и признаки после блока испорчены
        assert [r1[d]["f"] for d in ds] == [r2[d]["f"] for d in ds] and len({r1[d]["s2"] for d in ds}) == 1
    ok = d_fit.block_fc(F, sp, HP, Tm(), a, ds)
    bad = d_fit.block_fc(F, sp, HP, Tm(), ds[-1], ds)                              # мутация: якорь — последняя дата блока (видит цели до неё)
    assert [bad[d]["f"] for d in ds] != [ok[d]["f"] for d in ds]


def test_fold_window_invariant_to_rows_after_cut_and_mutation_caught(W):
    F, sp = W["F"], spec(W)
    Y, y = 2020, 2019      # год проверки рядом со срезом: окно из ближайших к нему пар захватывает испорченные строки при срезе на год позже
    T = select.cut_row(F, Y)
    dm = damaged(sp, T, T + 1)
    for tm in (Tm(win=504), Tm(hl=756.0)):
        a = d_fit.fold_q_d(F, sp, W["eps"], HP, tm, Y, y)[1].mean()
        assert a == d_fit.fold_q_d(F, dm, W["eps"], HP, tm, Y, y)[1].mean()
        assert d_fit.fold_q_d(F, dm, W["eps"], HP, tm, Y + 1, y)[1].mean() != d_fit.fold_q_d(F, sp, W["eps"], HP, tm, Y + 1, y)[1].mean()   # срез позже
