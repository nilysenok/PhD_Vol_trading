"""V5 лист 2, G19: тождества и известный ответ. Σ_b E_b = RV (в т. ч. n = 50 и 52), доли не зависят от n; M = Σ π exp(g) и M = 1 при θ = 0; аналитический градиент против численного; восстановление θ на синтетике с известным ответом (знак градиента перепутать нельзя);
SP1: при пропорциональных полосах R = 1; фактор SP3 не включает свою бумагу; SHA замороженных входов и K8 побитно."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import dct

from models_v2.s12 import v5_0_bars as BR, v5_0_ctrl as CT, v5_0_io as IO, v5_sp_feat as FE, v5_sp_fit as FI, v5_sp_har as HAR
from models_v2.s12.v5_gs_panel import Panel


def test_band_energies_sum_to_rv_for_n_50_and_52_and_mutation_dropped_bar_is_caught():
    rng = np.random.default_rng(1)
    for n in (50, 52):
        x = rng.normal(0, 0.003, n)
        s, lo, mi, hi = BR.band_energy(x)
        assert abs((lo + mi + hi) - s) / s < 1e-12 and abs(s - (x * x).sum()) < 1e-15
        s2, lo2, mi2, hi2 = BR.band_energy(x[:-1])                       # мутация: выброшен бар — тождество с исходным RV ломается
        assert abs((lo2 + mi2 + hi2) - s) / s > 1e-4


def test_shares_depend_on_k_over_n_not_on_n():
    for n, ks in ((50, (5, 25, 40)), (52, (5, 26, 42))):               # одна и та же доля k/n = 0,1 / 0,5 / 0,8 при разных n
        for k, band in zip(ks, (1, 2, 3)):
            e = np.zeros(n)
            e[k] = 1.0
            x = dct(e, type=3, norm="ortho")                           # вектор доходностей с энергией только в частоте k
            s, lo, mi, hi = BR.band_energy(x)
            got = np.array([lo, mi, hi]) / s
            assert np.isclose(got[band - 1], 1.0) and np.isclose(got.sum(), 1.0), (n, k)


def _zp(R=400, L=3, seed=0, pi=None):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1.0, (R, 3, L))
    p = np.full((R, 3), 1 / 3) if pi is None else pi
    return Z, p, np.exp(rng.normal(0, 0.3, R))


def test_mult_is_weighted_exp_and_one_at_zero_theta_exactly():
    Z, p, _ = _zp(pi=np.random.default_rng(2).dirichlet([3, 3, 3], 400))
    th = np.random.default_rng(3).normal(0, 0.3, (3, 3))
    g = np.einsum("rbl,bl->rb", Z, th)
    assert np.allclose(FI.mult(Z, p, th), (p * np.exp(g)).sum(axis=1), rtol=1e-12)
    assert np.array_equal(FI.mult(Z, p, np.zeros((3, 3))), np.ones(400))


def test_gradient_matches_numeric():
    Z, p, k8 = _zp(L=6, pi=np.random.default_rng(2).dirichlet([3, 3, 3], 400))
    rv = k8 * np.exp(np.random.default_rng(4).normal(0, 0.4, 400))
    s2 = np.maximum(((p[:, :, None] * Z) ** 2).mean(axis=0), 1e-12)
    th = np.random.default_rng(5).normal(0, 0.2, 18)
    v, g = FI.objective(th, Z, p, rv, k8, s2)
    num = np.array([(FI.objective(th + e, Z, p, rv, k8, s2)[0] - FI.objective(th - e, Z, p, rv, k8, s2)[0]) / 2e-6 for e in np.eye(18) * 1e-6])
    assert np.isfinite(v) and np.max(np.abs(g - num) / (np.abs(num) + 1e-8)) < 1e-6


def test_fit_recovers_known_theta_and_wrong_sign_gradient_would_not():
    rng = np.random.default_rng(7)
    R = 30000
    Z = rng.normal(0, 0.5, (R, 3, 3))
    p = np.full((R, 3), 1 / 3)
    k8 = np.exp(rng.normal(0, 0.3, R))
    true = np.array([[0.4, -0.2, 0.0], [0.0, 0.3, 0.0], [-0.3, 0.0, 0.2]])
    rv = k8 * FI.mult(Z, p, true) * rng.gamma(50.0, 1 / 50.0, R)         # шум со средним 1: QLIKE-оптимум — истинный прогноз
    th = FI.fit_theta(Z, p, rv, k8)
    assert np.max(np.abs(th - true)) < 0.08
    assert np.max(np.abs(FI.fit_theta(Z, p, rv, k8) + true)) > 0.2       # знак «минус» от истины далёк (перепутанный знак ловится)


def _panel_for_har(n=420, m=2, prop=True):
    rng = np.random.default_rng(11)
    ax = pd.bdate_range("2017-01-02", periods=n)
    D = pd.DatetimeIndex([ax[0] - pd.offsets.BDay(1)]).append(ax)
    rv = np.exp(rng.normal(0, 0.4, (n + 1, m)))
    w = np.array([0.2, 0.3, 0.5]) if prop else np.exp(rng.normal(0, 1, (n + 1, m, 3)))
    E = rv[..., None] * (w if prop else w / w.sum(-1, keepdims=True))
    mat = lambda a: np.full((n, m), np.nan) if a is None else a
    dd = np.broadcast_to(D[:-1].to_numpy()[:, None], (n, m)).copy()
    P = Panel(["A", "B"], ax, rv[1:], mat(None), mat(None), np.ones((n, m)), np.ones((n, m), bool), dd, np.broadcast_to((ax.year * 12 + ax.month).to_numpy()[:, None], (n, m)).astype(np.int64).copy())
    roll = lambda a, k: pd.DataFrame(a.reshape(len(a), -1)).rolling(k, min_periods=k).mean().to_numpy().reshape(a.shape)
    q = E / E.sum(-1, keepdims=True)
    F = FE.Feat(["A", "B"], D, E, rv, roll(E, 5), roll(E, 22), roll(rv, 5), roll(rv, 22), q, roll(q, 5), roll(q, 22), np.full(E.shape, 1 / 3), np.zeros(E.shape))
    return P, F


def test_sp1_proportional_bands_give_r_equal_one_and_shuffled_bands_do_not():
    P, F = _panel_for_har(prop=True)
    r = HAR.run(P, F)["M"]
    assert np.nanmax(np.abs(r - 1.0)) < 1e-6                              # Ê_b = c_b·Ê_tot, Σc_b = 1
    P2, F2 = _panel_for_har(prop=False)
    assert np.nanmax(np.abs(HAR.run(P2, F2)["M"] - 1.0)) > 1e-3


def test_sp1_r_is_clipped():
    P, F = _panel_for_har(prop=False)
    r = HAR.run(P, F)["M"]
    assert np.nanmin(r) >= HAR.LO and np.nanmax(r) <= HAR.HI


def test_cross_factor_excludes_own_ticker_and_excluded_ones():
    rng = np.random.default_rng(3)
    nD, m = 5, 4
    q = rng.dirichlet([2, 2, 2], (nD, m))
    F = FE.Feat(list("ABCD"), pd.bdate_range("2020-01-01", periods=nD), *(np.zeros((nD, m, 3)),) * 1, np.zeros((nD, m)), *(np.zeros((nD, m, 3)),) * 2, *(np.zeros((nD, m)),) * 2, q, q, q, np.full((nD, m, 3), 1 / 3), np.zeros((nD, m, 3)))
    f = FE.cross_factor(F, np.array([True, True, True, False]))
    want = np.log(q[:, [1, 2]].mean(axis=1)) - np.log(1 / 3)           # бумага A: другие в keep — B, C
    assert np.allclose(f[:, 0, :], want)
    f2 = FE.cross_factor(F, np.array([True, False, False, False]))
    assert np.all(f2[:, 0, :] == 0.0)                                   # других бумаг нет — 0


def test_frozen_inputs_sha_and_k8_bitwise_unchanged():
    assert bool(CT.sha_rows()["ok"].all())
    same, mx, nb, nout = CT.k8_rows(IO.frame())
    assert same and mx == 0.0 and nout == 0 and nb == len(IO.frame())
