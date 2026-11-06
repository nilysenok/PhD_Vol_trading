"""LGBM волна B: главный тест обрезки на реальном мире Б-2 — прогноз на d и фолд LOYO не зависят от строк после d / после 01.01.Y; мутации."""
import dataclasses

import numpy as np
import pytest

from models_v2.s4 import select
from models_v2.s6.lgbm import grid, job, st_cfgs, world

HP = grid.Hp(7, 100, 10.0, 0.05, 0.8, 1.0, -1, 150)
DATES = ("2018-06-01", "2022-06-01", "2025-06-02")


@pytest.fixture(scope="module")
def W():
    return world.load()


def specs(W):
    cols = st_cfgs.group_cols(W, sorted(W["acc"]))
    return {"ln": job.base_spec(W, "t_ln", "ln"), "qlike": job.base_spec(W, "t_q", "qlike"), "f_all": st_cfgs.feat_spec(W, "t_fa", cols)}


def damaged(sp, T):
    X, y = sp.X.copy(), sp.y.copy()
    X[T + 1:] = 1e6
    y[T:] = 1e6
    return dataclasses.replace(sp, X=X, y=y)


@pytest.mark.parametrize("which", ["ln", "qlike", "f_all"])
def test_day_forecast_invariant_to_rows_after_d(W, which):
    F, sp = W["F"], specs(W)[which]
    for ds in DATES:
        d = int(np.flatnonzero(F.dates == np.datetime64(ds))[0])
        a = job.day_fc(F, sp, HP, d)
        b = job.day_fc(F, damaged(sp, d), HP, d)
        assert a["f"] == b["f"] and a["k"] == b["k"]
        assert a["s2"] == b["s2"] or (np.isnan(a["s2"]) and np.isnan(b["s2"]))


def test_mutation_pairs_with_target_d_plus_1_is_caught(W):
    """Мутация: в обучение попадает пара с целью d+1 (знание завтрашнего дня) — прогноз меняется, значит тест обрезки чувствителен."""
    F, sp = W["F"], specs(W)["ln"]
    d = int(np.flatnonzero(F.dates == np.datetime64("2022-06-01"))[0])
    from models_v2.s6.lgbm import fitl
    ok = fitl.fit_l(sp.X, sp.y, job.pairs(F, sp, d), HP, "ln")["predict"](sp.X[[d]])[0]
    bad = fitl.fit_l(sp.X, sp.y, job.pairs(F, sp, d + 1), HP, "ln")["predict"](sp.X[[d]])[0]
    assert ok != bad


def test_fold_invariant_to_rows_after_cut(W):
    F, sp = W["F"], specs(W)["ln"]
    Y, y = 2020, 2018
    T = select.cut_row(F, Y)
    assert job.fold_q(F, sp, W["eps"], HP, Y, y) == job.fold_q(F, damaged(sp, T), W["eps"], HP, Y, y)


def test_mutation_fold_with_year_Y_target_leaks(W):
    """Мутация: порог выбора на год позже (цели Y включены) — QLIKE фолда меняется."""
    F, sp = W["F"], specs(W)["ln"]
    Y, y = 2020, 2018
    T = select.cut_row(F, Y)
    X, yy = sp.X.copy(), sp.y.copy()
    X[T + 1:] = 1e6
    yy[T:] = 1e6
    q_ok = job.fold_q(F, dataclasses.replace(sp, X=X, y=yy), W["eps"], HP, Y, y)
    q_leak = job.fold_q(F, dataclasses.replace(sp, X=X, y=yy), W["eps"], HP, Y + 1, y)   # T шире: попадают испорченные строки
    assert q_ok != q_leak
