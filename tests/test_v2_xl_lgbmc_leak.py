"""LGBM волна C: главный тест обрезки на реальном мире Б-2 + F4 для HP волны C (глубинно-ограниченные, λ1, min_gain, ff, bagging); мутации сдвига и среза."""
import dataclasses

import numpy as np
import pytest

from models_v2.s4 import select
from models_v2.s6.lgbm import c_hp, c_l0, fitl, job, world

HP = c_hp.HpC(8, 10, 2.0, 0.05, 0.7, 0.8, 3, 120, 42, l1=1.0, gain=1.0)
DATES = ("2018-06-01", "2022-06-01", "2025-06-02")


@pytest.fixture(scope="module")
def W():
    return world.load()


def spec(W):
    return c_l0.spec_of(W, "t_c", "F4")


def damaged(sp, T):
    X, y = sp.X.copy(), sp.y.copy()
    X[T + 1:] = 1e6
    y[T:] = 1e6
    return dataclasses.replace(sp, X=X, y=y)


def test_day_forecast_invariant_to_rows_after_d(W):
    F, sp = W["F"], spec(W)
    for ds in DATES:
        d = int(np.flatnonzero(F.dates == np.datetime64(ds))[0])
        a, b = job.day_fc(F, sp, HP, d), job.day_fc(F, damaged(sp, d), HP, d)
        assert a["f"] == b["f"] and a["k"] == b["k"] and a["s2"] == b["s2"]


def test_mutation_pairs_with_target_d_plus_1_is_caught(W):
    F, sp = W["F"], spec(W)
    d = int(np.flatnonzero(F.dates == np.datetime64("2022-06-01"))[0])
    ok = fitl.fit_l(sp.X, sp.y, job.pairs(F, sp, d), HP)["predict"](sp.X[[d]])[0]
    bad = fitl.fit_l(sp.X, sp.y, job.pairs(F, sp, d + 1), HP)["predict"](sp.X[[d]])[0]
    assert ok != bad


def test_fold_invariant_to_rows_after_cut_and_mutation_caught(W):
    F, sp = W["F"], spec(W)
    Y, y = 2020, 2018
    T = select.cut_row(F, Y)
    dm = damaged(sp, T)
    assert job.fold_q(F, sp, W["eps"], HP, Y, y) == job.fold_q(F, dm, W["eps"], HP, Y, y)
    assert job.fold_q(F, dm, W["eps"], HP, Y, y) != job.fold_q(F, dm, W["eps"], HP, Y + 1, y)    # срез на год позже захватывает испорченные строки


def test_base_hp_per_year_has_no_future_years(W):
    """HP года Y берутся из таблицы И2/LB0 по году Y — ключи покрывают 2017…2026 и не зависят от строк после среза."""
    assert sorted(c_l0.hp_of(W, "inh")) == list(range(2017, 2027)) and sorted(c_l0.hp_of(W, "own")) == list(range(2017, 2027))
