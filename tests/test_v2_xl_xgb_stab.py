"""B3 (Ж-206, волна B, поток XGB): численная устойчивость кастомного QLIKE — g/h против конечных разностей, сравнение с reg:gamma, признак нестабильной строки.
Мутации: знак градиента; гессиан без y; пол гессиана 1.0; отношение σ̂² вне [0,1; 10] не замечено."""
import numpy as np
import pytest

from models_v2.s4 import fit
from models_v2.s5.b1 import fitobj, losses
from models_v2.s6.xgb import stabx

HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8, 1.0, 150)
RNG = np.random.default_rng(11)


def synth(n=700, seed=3):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    mu = np.exp(0.6 * X[:, 0] - 0.3 * X[:, 1])
    return X, mu * r.gamma(4.0, 0.25, n)


def test_fd_error_on_declared_grid_within_tolerance():
    y, F = stabx.fd_grid()
    assert stabx.fd_error(y, F) <= stabx.TOL_FD


def test_mutation_sign_of_gradient_is_caught(monkeypatch):
    real = losses.grad_hess
    monkeypatch.setattr(losses, "grad_hess", lambda y, F: (-real(y, F)[0], real(y, F)[1]))
    assert stabx.fd_error(*stabx.fd_grid()) > stabx.TOL_FD


def test_mutation_hessian_without_y_is_caught(monkeypatch):
    real = losses.grad_hess
    monkeypatch.setattr(losses, "grad_hess", lambda y, F: (real(y, F)[0], np.exp(-np.asarray(F, float))))
    assert stabx.fd_error(*stabx.fd_grid()) > stabx.TOL_FD


def test_mutation_hessian_floor_one_is_caught(monkeypatch):
    monkeypatch.setattr(losses, "HESS_MIN", 1.0)
    assert stabx.fd_error(*stabx.fd_grid()) > stabx.TOL_FD


def fit_pair(X, y):
    idx = np.arange(len(y))
    return {l: fitobj.fit_loss(X, y, idx, HP, l) for l in ("qlike", "gamma")}


def test_gamma_and_qlike_forecasts_finite_positive_and_close_on_synthetic():
    X, y = synth()
    m = fit_pair(X, y)
    z = X[-100:]
    a, b = (np.exp(m[l]["predict"](z)) for l in ("qlike", "gamma"))
    assert np.isfinite(a).all() and (a > 0).all() and np.isfinite(b).all() and (b > 0).all()
    assert np.median(np.abs(np.log(a / b))) < 0.15       # одна и та же потеря при log-link; отличие — ранняя остановка
    for l in m:
        assert np.corrcoef(np.log(np.exp(m[l]["predict"](z))), np.log(y[-100:] + 1e-9))[0, 1] > 0.3   # модель не вырождена


def test_extreme_target_outlier_keeps_qlike_forecast_finite():
    X, y = synth()
    y = y.copy()
    y[100] *= 1e4                                         # экстремальная RV в обучении
    m = fitobj.fit_loss(X, y, np.arange(len(y)), HP, "qlike")
    p = np.exp(m["predict"](X[-50:]))
    assert np.isfinite(p).all() and (p > 0).all() and p.max() < 1e6


def test_unstable_flags_nonfinite_nonpositive_and_ratio():
    import pandas as pd
    ref = np.ones(5)
    ok = pd.DataFrame({"sigma2": [1.0, 2.0, 0.5, 1.2, 0.9]})
    assert not stabx.unstable(ok, ref)
    for bad in ([1, 2, np.nan, 1, 1], [1, 2, 0.0, 1, 1], [1, 2, -1.0, 1, 1], [1, 2, 11.0, 1, 1], [1, 2, 0.05, 1, 1]):
        assert stabx.unstable(pd.DataFrame({"sigma2": bad}), ref)


def test_stab_dates_first_per_year_and_feb_2022():
    from models_v2.s5.b2 import synth_b2
    from models_v2.s2 import features as ft
    F = ft.build(synth_b2.bars(700, 5))
    grid = np.arange(300, 699)
    ds = stabx.stab_dates(F, grid)
    yrs = F.dates[grid].astype("datetime64[Y]").astype(int) + 1970
    assert len(ds) == len(set(yrs)) and all(d in set(grid.tolist()) for d in ds)
    firsts = [int(grid[np.flatnonzero(yrs == y)[0]]) for y in sorted(set(yrs))]
    assert ds == firsts                                   # синтетика кончается в 2021 — даты 22.02.2022 нет, лишних дат нет
