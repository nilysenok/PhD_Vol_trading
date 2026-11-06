"""Ж-212 (HS7, G7, G10, G14): Component Hybrid на синтетике с известным ответом: восстановление коэффициентов D0, нелинейная форма, положительность, обрезка; мутация «будущие пары в подгонке» ловится."""
from __future__ import annotations

import numpy as np

from models_v2.s4 import fit
from models_v2.s8 import hs_dfit as D, hs_leak
from tests.test_v2_hs_feat import SynF  # noqa: E402  (синтетика PIT-тестов)


def synth_xy(n=4000, seed=1, shape=40.0):
    rng = np.random.default_rng(seed)
    Phi = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    ln_f = 0.2 + 0.5 * Phi[:, 0] + 0.3 * Phi[:, 2]
    y = np.exp(ln_f) * rng.gamma(shape, 1 / shape, n)
    return Phi, np.append(y[1:], np.nan), np.arange(n)


def test_lin_fit_recovers_known_coefficients_and_drop_changes_loss():
    Phi, _, _ = synth_xy()
    y = np.exp(0.2 + 0.5 * Phi[:, 0] + 0.3 * Phi[:, 2]) * np.random.default_rng(2).gamma(40.0, 1 / 40.0, len(Phi))
    idx = np.arange(len(Phi))
    g = D.lin_fit(Phi, y, idx, range(5))
    assert abs(g[0] - 0.2) < 0.05 and abs(g[1] - 0.5) < 0.05 and abs(g[3] - 0.3) < 0.05 and np.abs(g[[2, 4, 5]]).max() < 0.05
    full, drop = D.lin_fit(Phi, y, idx, range(5)), D.lin_fit(Phi, y, idx, (1, 2, 3, 4))      # без информативной φ_0
    from models_v2.s8 import hs_qfit
    lf = lambda gm, cols: hs_qfit.mean_loss(gm, np.column_stack([np.ones(len(idx)), Phi[:, cols]]), y, np.zeros(len(idx)))
    assert lf(drop, [1, 2, 3, 4]) - lf(full, [0, 1, 2, 3, 4]) > 0.05


def test_nonlinear_shape_is_recovered_by_stumps_and_prediction_is_positive():
    rng = np.random.default_rng(5)
    n = 3000
    Phi = rng.normal(0, 1, (n, 5))
    y = np.exp(0.1 + 0.4 * Phi[:, 2] + 0.8 * (Phi[:, 0] > 0)) * rng.gamma(60.0, 1 / 60.0, n)
    ln_s2, shape = D.nl_day(Phi, y, np.zeros(n, int), Phi[:, [0]], 0, 1, n - 2, np.array([-1.5, 1.5]))
    assert np.isfinite(ln_s2) and 0.5 < shape[1] - shape[0] < 1.0       # истинный скачок 0,8; форма читается из пней
    assert np.exp(ln_s2) > 0


def ctx_of(F):
    return {"w": {"F": F, "eps": np.full(len(F.rv), 1e-6)}, "ml": {"XGB": np.ones(len(F.rv))}}


def test_truncation_linear_and_nonlinear_days_unchanged():
    F = SynF(500)
    for d in (330, 420):
        assert hs_leak.trunc_d(ctx_of(F), d, cols=(0, 1, 2, 3, 4)) == 0.0
        assert hs_leak.trunc_d(ctx_of(F), d, k=0) == 0.0


def test_x_state_terciles_use_only_past_and_whole_sample_mutation_is_caught():
    from models_v2.s5 import pit
    from models_v2.s8 import hs_dstage
    F = SynF(500)
    mk = np.exp(np.random.default_rng(4).normal(0, 0.5, 500))
    fn = hs_dstage.x_state(mk)
    d = 400
    Fc, _ = hs_leak.corrupt(F, {"XGB": np.ones(500)}, d)
    a, b = fn(F, F.X[:, 0]), fn(Fc, Fc.X[:, 0])
    assert np.array_equal(a[:d + 1, 1:], b[:d + 1, 1:], equal_nan=True)
    whole = lambda Fx: np.digitize(np.exp(Fx.X[:, 1]), np.nanquantile(np.exp(Fx.X[:, 1]), [1 / 3, 2 / 3]))   # мутация: пороги по всей выборке
    assert not np.array_equal(whole(F)[:d + 1], whole(Fc)[:d + 1])
    assert pit.past_bucket is not None


def test_mutation_future_pairs_in_fit_are_caught():
    F = SynF(500)
    d = 420
    Fc, _ = hs_leak.corrupt(F, {"XGB": np.ones(500)}, d)
    eps = np.full(500, 1e-6)

    def day(Fx, extra):
        Phi, y = D.phi_matrix(Fx), D.target(Fx, eps)
        idx = fit.pair_index(Fx.seg, Phi, y, tuple(range(5)), d + extra)      # extra=1: пара (d, RV_{d+1}) — будущее
        return D.lin_fit(Phi, y, idx, range(5))[0]
    assert day(F, 0) == day(Fc, 0) and day(F, 1) != day(Fc, 1)
