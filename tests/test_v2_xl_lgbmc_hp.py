"""LGBM волна C: HP с lambda_l1 / min_gain_to_split доходят до LightGBM; шаги осей C-L1 — известные ответы; мутации."""
import numpy as np
import pytest

from models_v2.s6.lgbm import c_hp, fitl, grid

H = grid.Hp(7, 30, 10.0, 0.02, 0.8, 0.8)


def test_params_carry_l1_and_gain():
    p = c_hp.lift(H).params()
    assert p["lambda_l1"] == 0.0 and p["min_gain_to_split"] == 0.0
    p = c_hp.HpC(7, 30, 10.0, 0.02, 0.8, 0.8, l1=5.0, gain=10.0).params()
    assert p["lambda_l1"] == 5.0 and p["min_gain_to_split"] == 10.0
    assert p["bagging_freq"] == 1 and c_hp.HpC(7, 30, 10.0, 0.02, 0.8, 1.0).params()["bagging_freq"] == 0


def test_label_distinguishes_l1_and_gain_and_matches_base_label():
    assert c_hp.lift(H).label() == H.label()
    assert len({c_hp.HpC(7, 30, 10.0, 0.02, 0.8, 0.8, l1=l, gain=g).label() for l in (0.0, 1.0) for g in (0.0, 1.0)}) == 4


def test_axis_steps_known_answers():
    a = c_hp.AXES
    assert [h.num_leaves for h in a["num_leaves"](H)] == [4, 14]
    assert [h.min_data for h in a["min_data"](H)] == [15, 60]
    assert [h.min_data for h in a["min_data"](grid.Hp(7, 8, 10.0, 0.02, 0.8, 0.8))] == [5, 16]       # нижняя граница 5
    assert [h.lr for h in a["lr"](H)] == [0.01, 0.04]
    assert [h.l2 for h in a["l2"](H)] == [pytest.approx(10 / 3, abs=1e-5), 30.0]
    assert [h.ff for h in a["ff"](H)] == [0.6, 1.0] and [h.ff for h in a["ff"](grid.Hp(7, 30, 2.0, 0.02, 0.5, 1.0))] == [0.3, 0.7]
    assert [h.bag for h in a["bag"](H)] == [0.6, 1.0]
    assert [h.max_depth for h in a["max_depth"](H)] == [3, 6]
    D = grid.Hp(8, 10, 2.0, 0.05, 1.0, 0.8, 3)                      # глубинно-ограниченная база (как И2)
    assert [(h.num_leaves, h.max_depth) for h in a["num_leaves"](D)] == [(4, 3), (16, 4)]   # hi поднимает глубину, иначе листья не добавятся
    assert [(h.num_leaves, h.max_depth) for h in a["max_depth"](D)] == [(8, 2), (8, 4)]
    assert [h.max_depth for h in a["max_depth"](grid.Hp(4, 10, 2.0, 0.05, 1.0, 0.8, 2))] == [2, 3]
    assert [(h.l1) for h in a["l1"](H)] == [1.0, 5.0] and [h.gain for h in a["gain"](H)] == [1.0, 10.0]


def test_move_applies_to_each_year_and_sides_differ():
    hp = {2017: H, 2018: grid.Hp(31, 200, 2.0, 0.05, 0.5, 1.0)}
    lo, hi = c_hp.move(hp, "num_leaves", 0), c_hp.move(hp, "num_leaves", 1)
    assert [h.num_leaves for h in lo.values()] == [4, 16] and [h.num_leaves for h in hi.values()] == [14, 62]
    assert c_hp.changed_axes(lo, hp) == ["num_leaves"] and c_hp.changed_axes(hp, hp) == []


def _toy(n=900):
    r = np.random.default_rng(0)
    X = r.normal(size=(n, 5))
    return X, X[:, 0] * 0.5 + r.normal(scale=0.3, size=n), np.arange(n)


def test_l1_changes_the_fitted_model():
    """Мутация «lambda_l1 не передан в LightGBM» -> прогнозы совпали бы; тест требует различия."""
    X, y, idx = _toy()
    a = fitl.fit_l(X, y, idx, c_hp.HpC(7, 30, 2.0, 0.05, 1.0, 1.0, max_trees=60))["predict"](X[:20])
    b = fitl.fit_l(X, y, idx, c_hp.HpC(7, 30, 2.0, 0.05, 1.0, 1.0, max_trees=60, l1=50.0))["predict"](X[:20])
    assert not np.allclose(a, b)


def test_huge_min_gain_gives_constant_prediction():
    X, y, idx = _toy()
    m = fitl.fit_l(X, y, idx, c_hp.HpC(7, 30, 2.0, 0.05, 1.0, 1.0, max_trees=60, gain=1e9))["predict"](X[:50])
    assert np.ptp(m) < 1e-9
    free = fitl.fit_l(X, y, idx, c_hp.HpC(7, 30, 2.0, 0.05, 1.0, 1.0, max_trees=60))["predict"](X[:50])
    assert np.ptp(free) > 1e-3
