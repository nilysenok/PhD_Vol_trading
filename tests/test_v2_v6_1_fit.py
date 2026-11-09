"""V6-1a, G12: обучатели LIN и GBM. Градиент и гессиан QLIKE по g = численным (≤ 1e-6); LIN на синтетике с известной линейной g восстанавливает θ (≤ 0,05); λ — из сетки плана по закрытым парам;
GBM: ранняя остановка на последних 252 парах, затем переобучение на всех с этим числом деревьев; clip |g| ≤ 1; g ≡ 0 → F = H побитно. Мутации № 99 (блок остановки в обучении) и № 100 (финал без последних 252 пар) ловятся."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s13 import v6_1_fit as FT


def synth(n: int, seed: int = 0, nonlinear: bool = False):
    """e = g_true + ln ε, ε ~ Gamma(5, 1/5): E ε = 1, поэтому g_true — минимум QLIKE."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * np.array([1.0, 2.0, 0.5]) + np.array([0.5, -1.0, 3.0])
    g = np.where(X[:, 0] > 0.5, 0.4, -0.3) if nonlinear else 0.2 + 0.3 * X[:, 0] / 1.0 - 0.1 * X[:, 1]
    return X, g, g + np.log(rng.gamma(5.0, 0.2, n))


def test_grad_hess_equal_numeric():
    rng = np.random.default_rng(1)
    e, g = rng.normal(0, 0.5, 50), rng.normal(0, 0.5, 50)
    h = 1e-5
    num_g = (FT.qlike_g(e, g + h) - FT.qlike_g(e, g - h)) / (2 * h)
    gr, he = FT.grad_hess(e, g)
    num_h = (FT.grad_hess(e, g + h)[0] - FT.grad_hess(e, g - h)[0]) / (2 * h)
    assert np.abs(gr - num_g).max() <= 1e-6 and np.abs(he - num_h).max() <= 1e-6


def test_lin_recovers_known_linear_g():
    X, g, e = synth(4000)
    m = FT.fit_lin(X, e)
    assert np.abs(m.g(X) - g).mean() <= 0.02                                                          # на хвостах X (до 4σ) ошибка наклона 0,01–0,02 даёт до 0,07: критерий по θ ниже
    sd, mu = np.array([1.0, 2.0, 0.5]), np.array([0.5, -1.0, 3.0])                                 # θ — в координатах z = (x − μ)/σ плана
    true = np.r_[0.2 + 0.3 * mu[0] - 0.1 * mu[1], 0.3 * sd[0], -0.1 * sd[1], 0.0]
    assert np.abs(m.theta - true).max() <= 0.05


def test_lambda_from_grid_and_selected_by_holdout_scores():
    X, g, e = synth(900, seed=2)
    m = FT.fit_lin(X, e)
    i, sc = FT.select_lambda(X, e)
    assert m.lam in FT.GRID and len(sc) == len(FT.GRID) and i == int(np.argmin(sc)) and m.lam == FT.GRID[i]
    assert FT.fit_lin(X, e, lam_shift=1).lam == FT.GRID[min(i + 1, 3)] and FT.fit_lin(X, e, lam_shift=-1).lam == FT.GRID[max(i - 1, 0)]


def test_lambda_selection_uses_only_given_pairs_last_252_are_holdout():
    X, g, e = synth(700, seed=4)
    _, sc = FT.select_lambda(X, e)
    e2 = e.copy()
    e2[448:] = e2[448:][::-1]                                         # меняется только блок проверки (последние 252 пары): оценки λ обязаны измениться
    _, sc2 = FT.select_lambda(X, e2)
    assert sc != sc2
    e3 = e.copy()
    e3[:448] = e3[:448][::-1]                                         # меняется только обучение: оценки тоже меняются (проверка идёт по параметрам обучения)
    assert FT.select_lambda(X, e3)[1] != sc
    tr, va = FT.split_es(700)
    assert len(va) == 252 and tr.max() < va.min()


def test_too_few_pairs_refused():
    X, g, e = synth(100)
    with pytest.raises(ValueError):
        FT.fit_lin(X, e)
    with pytest.raises(ValueError):
        FT.fit_gbm(X, e)


def test_clip_and_zero_g_gives_h_bitwise():
    h = np.array([1.3, 0.7, 2.2])
    assert np.array_equal(FT.forecast(h, np.zeros(3)), h)
    assert np.allclose(FT.forecast(h, np.array([9.0, -9.0, 0.5])), h * np.exp([1.0, -1.0, 0.5]))
    assert (FT.forecast(h, np.array([-50.0, 50.0, 0.0])) > 0).all()


def gbm_ok(info: dict, n: int) -> bool:
    """Инварианты G12: обучение остановки строго до блока проверки; блок = 252; финал — на всех n парах."""
    return info["train_max"] < info["es_min"] and info["n_es"] == 252 and info["n_final"] == n and info["final_max"] == n - 1


def test_gbm_early_stop_then_refit_on_all():
    X, g, e = synth(1500, seed=5, nonlinear=True)
    m = FT.fit_gbm(X, e)
    assert gbm_ok(m.info, 1500) and 1 <= m.n_trees <= FT.MAX_TREES and m.info["n_train"] == 1500 - 252
    Xn, gn, en = synth(1500, seed=6, nonlinear=True)
    assert FT.qlike_g(en, FT.clip_g(m.g(Xn))).mean() < FT.qlike_g(en, 0.0).mean()
    assert np.array_equal(m.g(Xn), FT.fit_gbm(X, e).g(Xn))                                            # детерминизм
    assert FT.fit_gbm(X, e, trees_mult=0.5).n_trees == max(1, int(round(m.info["best"] * 0.5)))


def test_mutation_99_es_block_in_training_caught(monkeypatch):
    X, g, e = synth(1500, seed=5, nonlinear=True)
    monkeypatch.setattr(FT, "split_es", lambda n, hold=FT.HOLD: (np.arange(n), np.arange(n - hold, n)))
    assert not gbm_ok(FT.fit_gbm(X, e).info, 1500)


def test_mutation_100_final_without_last_252_caught(monkeypatch):
    X, g, e = synth(1500, seed=5, nonlinear=True)
    monkeypatch.setattr(FT, "final_idx", lambda n, hold=FT.HOLD: np.arange(n - hold))
    assert not gbm_ok(FT.fit_gbm(X, e).info, 1500)
