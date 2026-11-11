"""Ж-257 п. 3: контроль S-RL1b2 — π на 252 строки назад БЕЗ переноса; не использует π с индексом > i; первые 252 пары вне пула обучения; мутация `roll` ловится."""
import numpy as np
import pytest

from models_v2.s18 import l_model as ML
from models_v2.s18 import l_reg2 as R2

N = 2198


def data(seed=5, n=N):
    rng = np.random.default_rng(seed)
    return dict(rv=np.exp(rng.normal(size=n) * 0.4), lnb=rng.normal(size=n) * 0.2, pi=rng.uniform(0.1, 0.9, n), bg=np.zeros(n), lnc8=rng.normal(size=n) * 0.1)


def test_pi_of_row_i_is_pi_i_minus_252_and_never_uses_index_above_i():
    pi = np.random.default_rng(1).uniform(size=N)
    for i in (252, 253, 700, 2197):
        out = ML.train_pi("shift252nr", pi, np.array([i]))[0]
        assert out == pi[i - 252]
        p2 = pi.copy()
        p2[i - 251:] = np.random.default_rng(2).uniform(size=N - i + 251)    # будущее относительно i − 252, включая i и дальше
        assert ML.train_pi("shift252nr", p2, np.array([i]))[0] == out


def test_first_252_rows_have_no_past_value_and_are_dropped_not_filled():
    pi = np.random.default_rng(1).uniform(size=N)
    with pytest.raises(ValueError):
        ML.train_pi("shift252nr", pi, np.arange(0, 300))
    assert np.array_equal(ML.train_idx("shift252nr", np.arange(0, 300)), np.arange(252, 300)) and np.array_equal(ML.train_idx("shift252", np.arange(0, 300)), np.arange(0, 300))
    A, T = data(), np.arange(0, 900)
    assert ML.stats_l("shift252nr", A, T) == ML.stats_l("shift252nr", A, np.arange(252, 900))
    assert ML.fit_kind("shift252nr", A, T, 1e-2, 3.0, 1.0) == ML.fit_kind("shift252nr", A, np.arange(252, 900), 1e-2, 3.0, 1.0)


def test_mutation_roll_without_exclusion_is_caught_and_differs_from_cyclic_control():
    A, T = data(), np.arange(0, 900)
    roll_all = lambda: ML.stats_l("shift252", A, T)    # мутация: циклический сдвиг, первые 252 берут π из конца истории
    assert roll_all() != ML.stats_l("shift252nr", A, T)
    pi = A["pi"]
    leaked = ML.train_pi("shift252", pi, np.arange(0, 252))
    assert (leaked == pi[N - 252:]).all()    # будущее в начале истории — то, что запрещает Ж-257


def test_registry_row_marked_and_n_plus_one():
    assert R2.ADD == 1 and R2.ADD_WIDE == 2 and R2.IDS == ("S-RL1b2",) and "Ж-257 п. 3 до счёта" in R2.ROWS[0][4] and "заменяет циклический S-RL1b" in R2.ROWS[0][4]
    assert R2.check() == []
