"""И2 (Ж-200): сетка 24, ничья, PIT-срез Y, LOYO без целевого года; мутации M4, M5 ловятся."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s2 import features as ft
from models_v2.s4 import design, fit, runner, select, synth


@pytest.fixture(scope="module")
def F():
    return ft.build(synth.random_bars(1150, 3))     # 2019-01 … 2023-06


def test_grid_is_24_distinct_with_frozen_subsample():
    g = select.GRID
    assert len(g) == len(set(g)) == 24 and {h.subsample for h in g} == {0.8}
    assert {h.depth for h in g} == {2, 3, 5} and {h.min_child for h in g} == {10, 30}
    assert {h.l2 for h in g} == {0.5, 2.0} and {h.lr for h in g} == {0.05, 0.1}


def test_pick_known_best_and_tie_order():
    sc = {h: 1.0 for h in select.GRID}
    best = select.GRID[7]
    sc[best] = 0.5
    assert select.pick(sc, select.order_key) == best
    flat = {h: 0.3 for h in select.GRID}
    assert select.pick(flat, select.order_key) == fit.HP(2, 30, 2.0, 0.05, 0.8)   # меньше глубина, больше min_data, больше L2, меньше lr
    flat[select.GRID[3]] += 1e-13                                                   # в пределах допуска ничьей
    assert select.pick(flat, select.order_key) == fit.HP(2, 30, 2.0, 0.05, 0.8)
    assert select.pick({h: np.nan for h in select.GRID}) is None


def test_cut_row_is_last_row_before_january_first(F):
    c = select.cut_row(F, 2021)
    assert F.dates[c] < np.datetime64("2021-01-01") <= F.dates[c + 1]


def test_pit_pairs_targets_before_year_start_and_mutation_m5(F):
    dz = design.make(F, "level")
    Y = 2022
    T = select.cut_row(F, Y)

    def leaks(T_cut):
        return F.dates[runner.pairs_upto(dz, F, T_cut) + 1].max() >= np.datetime64("%d-01-01" % Y)
    assert not leaks(T) and leaks(T + 5)         # M5: срез позже 01.01.Y виден


def test_fold_split_excludes_target_year_and_mutation_m4(F):
    dz = design.make(F, "level")
    T = select.cut_row(F, 2022)
    tr, val = runner.fold_split(F, dz, T, 2020)
    yr = lambda a: pd.DatetimeIndex(F.dates[a + 1]).year  # noqa: E731
    assert set(yr(val)) == {2020} and 2020 not in set(yr(tr)) and len(set(tr) & set(val)) == 0
    assert max(yr(tr)) == 2021                    # обучение видит и годы после y (внутри прошлого)
    bad = np.concatenate([tr, val])               # M4: целевой год в обучении
    assert 2020 in set(yr(bad))


def test_fold_score_finite_and_deterministic(F):
    dz = design.make(F, "level")
    hp = fit.HP(3, 10.0, 2.0, 0.1, 0.8)
    a = runner.fold_score(F, dz, hp, select.cut_row(F, 2022), 2021, False)
    assert np.isfinite(a) and a == runner.fold_score(F, dz, hp, select.cut_row(F, 2022), 2021, False)
    assert np.isnan(runner.fold_score(F, dz, hp, select.cut_row(F, 2019) + 30, 2015, False))   # нет пар года -> NaN
