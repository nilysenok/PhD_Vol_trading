"""V7-1a-E, G15: гейт новизны E0. Известный ответ на синтетике: z — линейная функция опорных признаков ⇒ R² ≈ 1; независимый z ⇒ R² ≈ 0.
PIT: прогноз дня d не зависит от строк с датами ≥ первого дня его месяца. Мутация: регрессия по всей выборке — ловится."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_1e_e0 as E0

N = 1600
DATES = pd.bdate_range("2016-01-04", periods=N)


def ref(seed=0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(N, 4)), 0) * 0.1 + rng.normal(size=(N, 4))      # S, Σ, ln H, ℓ: коррелированные, с трендом
    return X


def lin(X, seed=1, noise=0.01):
    rng = np.random.default_rng(seed)
    return np.einsum("ij,jk->ik", X, rng.normal(size=(4, 4))) + noise * rng.normal(size=(N, 4))


def test_duplicate_of_reference_gives_r2_near_one_and_blocks_encoder():
    X = ref()
    g = E0.novelty_gate(lin(X), X, DATES)
    assert (g["r2"] > 0.99).all() and g["no_new_info"] is True and g["n_eval"] > 900


def test_independent_z_gives_r2_near_zero_and_passes_gate():
    X = ref()
    z = np.random.default_rng(5).normal(size=(N, 4))
    g = E0.novelty_gate(z, X, DATES)
    assert (np.abs(g["r2"]) < 0.05).all() and g["no_new_info"] is False


def test_all_components_required_for_stop():
    X = ref()
    z = lin(X)
    z[:, 2:] = np.random.default_rng(6).normal(size=(N, 2))
    g = E0.novelty_gate(z, X, DATES)
    assert (g["r2"][:2] > 0.99).all() and (np.abs(g["r2"][2:]) < 0.05).all() and g["no_new_info"] is False


def test_partial_explanation_known_answer():
    """z = a·(ref) + b·(независимое): R² ≈ a²var/(a²var + b²var)."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(N, 4))
    z = np.stack([x[:, 0] + rng.normal(size=N)] * 1, 1)        # доля объяснённой дисперсии 1/2
    g = E0.novelty_gate(z, x, DATES)
    assert abs(g["r2"][0] - 0.5) < 0.06


def test_window_and_threshold_parameters():
    X = ref()
    g = E0.novelty_gate(lin(X), X, DATES, start="2018-01-01", end="2018-12-31")
    ev = ~np.isnan(g["pred"][:, 0])
    d = DATES[ev]
    assert d.min() >= pd.Timestamp(2018, 1, 1) and d.max() <= pd.Timestamp(2018, 12, 31)
    assert E0.novelty_gate(lin(X), X, DATES, thr=1.5)["no_new_info"] is False
    with pytest.raises(ValueError):
        E0.novelty_gate(lin(X), X, DATES[::-1])
    with pytest.raises(ValueError):
        E0.novelty_gate(lin(X)[:50], X[:50], DATES[:50])       # нет окна оценки


def changed_future(X, z, t):
    X2, z2 = X.copy(), z.copy()
    f = DATES >= t
    rng = np.random.default_rng(11)
    X2[f] = rng.normal(size=X2[f].shape) * 50
    z2[f] = rng.normal(size=z2[f].shape) * 50
    return X2, z2


def test_prediction_for_month_ignores_rows_from_its_first_day():
    X = ref()
    z = np.random.default_rng(5).normal(size=(N, 4))
    t = pd.Timestamp(2019, 6, 3)                                # первый рабочий день месяца
    X2, z2 = changed_future(X, z, t)
    a, b = E0.novelty_gate(z, X, DATES)["pred"], E0.novelty_gate(z2, X2, DATES)["pred"]
    before = DATES < t
    assert np.array_equal(a[before], b[before], equal_nan=True) and not np.array_equal(a[~before], b[~before], equal_nan=True)


def test_mutation_fit_on_whole_sample_caught(monkeypatch):
    X = ref()
    z = np.random.default_rng(5).normal(size=(N, 4))
    t = pd.Timestamp(2019, 6, 3)
    X2, z2 = changed_future(X, z, t)
    orig = E0._ridge_pred

    def whole(full_X, full_z):
        return lambda Xp, zp, Xe, lam: orig(full_X, full_z, Xe, lam)
    monkeypatch.setattr(E0, "_ridge_pred", whole(X, z))
    a = E0.novelty_gate(z, X, DATES)["pred"]
    monkeypatch.setattr(E0, "_ridge_pred", whole(X2, z2))
    b = E0.novelty_gate(z2, X2, DATES)["pred"]
    before = DATES < t
    assert not np.array_equal(a[before], b[before], equal_nan=True)
