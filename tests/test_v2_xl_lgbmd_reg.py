"""LGBM волна D: причинные режимные признаки, переключение по флагу своего дня, метрики переходов (MD4, MD5)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import fit
from models_v2.s6.lgbm import c_l0, d_comp, d_reg, world


def test_quintile_pit_and_mutation_caught():
    inc, dec = np.arange(100.0), np.arange(100.0)[::-1]
    assert np.isnan(d_reg.quintile_pit(inc)[:59]).all() and (d_reg.quintile_pit(inc)[70:] == 4).all()    # растёт — верхний квинтиль
    assert (d_reg.quintile_pit(dec)[70:] == 0).all()                                                       # падает — нижний
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    x2 = x.copy()
    x2[201:] = rng.normal(size=99) * 50
    assert np.array_equal(d_reg.quintile_pit(x)[:201], d_reg.quintile_pit(x2)[:201], equal_nan=True)       # будущее не влияет
    full = np.array([min(4, int(5 * (x < v).mean())) for v in x2], float)                                  # мутация: ранг по всей выборке
    assert not np.array_equal(full[:201], d_reg.quintile_pit(x2)[:201])


def test_transition_pit_and_mutation_caught():
    st = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], float)
    tr =d_reg.transition_pit(st)
    assert np.isnan(tr[0]) or tr[0] == 0
    assert tr[2:7].tolist() == [1, 1, 1, 1, 1] and tr[7] == 0 and tr[8:].tolist() == [1, 1]
    st2 = st.copy()
    st2[6:] = 0
    assert np.array_equal(d_reg.transition_pit(st2)[:6], tr[:6], equal_nan=True)                           # строки после t не влияют
    shifted = np.r_[tr[1:], np.nan]                                                                        # мутация: сдвиг на +1 (признак «из завтра»)
    assert not np.array_equal(shifted[:7], tr[:7], equal_nan=True)
    assert np.isnan(d_reg.transition_pit(np.array([np.nan, np.nan, 0, 0.0]))[:2]).all()


def test_flip_on_grid_known_answer():
    st = np.array([np.nan, 0, 0, 1, 1, 0], float)
    assert d_reg.flip_on_grid(st, np.arange(6)).tolist() == [False, False, False, True, False, True]
    assert d_reg.flip_on_grid(st, np.array([1, 3, 4])).tolist() == [False, True, False]                    # сетка с пропусками строк


def test_switch_uses_flag_of_own_day_and_mutation_caught():
    one = pd.DataFrame({"sigma2": [1.0] * 4, "rv": [1.0] * 4})
    two = pd.DataFrame({"sigma2": [2.0] * 4, "rv": [1.0] * 4})
    fcs = {"A": one, "B": two}
    st = np.array([0, 1, np.nan, 1.0])
    yrs = np.array([2017] * 4)
    fc, act = d_comp.switch_series(fcs, {2017: "A"}, {2017: "B"}, st, yrs)
    assert fc["sigma2"].tolist() == [1, 2, 1, 2] and act.tolist() == ["A", "B", "A", "B"]                   # NaN режима — спокойно
    bad, _ = d_comp.switch_series(fcs, {2017: "A"}, {2017: "B"}, np.r_[st[1:], 0.0], yrs)                  # мутация: флаг завтрашнего дня
    assert bad["sigma2"].tolist() != fc["sigma2"].tolist()


def test_switch_folds_and_pick_by_regime_known_answer():
    mk = lambda q: pd.DataFrame({"Y": 2018, "y": 2017, "row": [0, 1, 2, 3], "q": q})
    dfs = {"A": mk([1.0, 1, 1, 1]), "B": mk([2.0, 2, 2, 2]), "C": mk([3.0, 0, 3, 0])}
    st = np.array([0, 1, 0, 1.0])
    sc = d_comp.regime_scores(dfs, st)
    assert sc[(2018, 0)].to_dict() == {"A": 1.0, "B": 2.0, "C": 3.0} and sc[(2018, 1)].to_dict() == {"A": 1.0, "B": 2.0, "C": 0.0}
    calm, stress = d_comp.pick_by_regime(sc, ["A", "B", "C"], [2018, 2019])
    assert calm == {2018: "A", 2019: "A"} and stress == {2018: "C", 2019: "A"}                            # нет фолдов года -> первое имя
    sf = d_comp.switch_folds(dfs, {2018: "A"}, {2018: "C"}, st)
    assert sf["q"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_transitions_known_answer():
    st = np.array([0, 0, 1, 1, 0, 0, 0, 0], float)
    fc = pd.DataFrame({"sigma2": [1, 1, 0.4, 0.6, 1, 1, 1, 1.0], "rv": [1.0] * 8})
    t = d_comp.transitions(fc, st)
    assert t["n_onsets"] == 1 and t["n_exits"] == 1 and t["delay_median"] == 1.0                           # σ̂²/RV: 0,4 в день входа, 0,6 на следующий
    assert t["under_on5"] == pytest.approx(2 / 5)                                                          # строки 2…6: заниженны строки 2 и 3 (0,4 и 0,6 < RV = 1)


@pytest.fixture(scope="module")
def W():
    return world.load()


def test_regime_columns_real_series_pit_and_pairs_unchanged(W):
    F = W["F"]
    st = d_reg.stress_rows(F.dates)
    g = W["grid"]
    assert np.isfinite(st[g]).all() and np.isnan(st[:g[0]]).all()                                          # ряд p3 — с первой даты сетки, раньше NaN, не «True»
    sp = c_l0.spec_of(W, "t_d", "F4")
    sp2 = d_reg.with_regime(sp, F.dates, d_reg.R1["R1d_all"], "t_d_r1")
    assert sp2.X.shape[1] == sp.X.shape[1] + 3 and np.array_equal(sp2.X[:, :sp.X.shape[1]], sp.X, equal_nan=True)
    T = int(g[500])
    assert np.array_equal(fit.pair_index(F.seg, sp.X, sp.y, (0, 1, 2, 3), T), fit.pair_index(F.seg, sp2.X, sp2.y, (0, 1, 2, 3), T))
    q5 = sp2.X[:, sp.X.shape[1] + 1]
    assert np.nanmin(q5) == 0 and np.nanmax(q5) == 4
