"""V4-C лист C-b (G19): оценщики C5, C6, C8, C9, C10 на синтетике с известным ответом и тождества. Слабые штрафы (kappa, lam) там, где проверяется восстановление параметров."""
import numpy as np
import pytest

from models_v2.s11 import v4_cb_fit as FB, v4_cb_models as MD, v4_c_fit as CF


def gate_data(n=20000, seed=3, gamma=1.0, k=3):
    """ell ~ U(-1,1)·0,8; z (k признаков N(0,1)); истинный гейт w = σ(0,2 + gamma·z[:,2]); rv = F·(шум с E=1)."""
    rng = np.random.default_rng(seed)
    ell = rng.uniform(-0.8, 0.8, n)
    lH = rng.normal(0, 0.5, n)
    Z = rng.normal(0, 1, (n, k))
    w = 1 / (1 + np.exp(-(0.2 + gamma * Z[:, -1])))
    F = np.exp(lH + w * ell)
    rv = F * rng.gamma(40.0, 1 / 40.0, n)
    return rv, lH, ell, Z, np.zeros(n, int), w


def test_c6_known_monotone_gate_gamma_sign_recovered():
    rv, lH, ell, Z, tid, _ = gate_data(gamma=1.2)
    r = FB.fit_lin(rv, lH, ell, Z, tid, 1, nonneg=(2,), kappa=1e-6, lam=1e-6)
    assert r["beta"][2] > 0.3 and r["ok"]


def test_c6_constraint_active_when_true_gamma_negative_and_mutation_caught():
    rv, lH, ell, Z, tid, _ = gate_data(gamma=-1.2)
    honest = FB.fit_lin(rv, lH, ell, Z, tid, 1, nonneg=(2,), kappa=1e-6, lam=1e-6)
    mutant = FB.fit_lin(rv, lH, ell, Z, tid, 1, nonneg=(), kappa=1e-6, lam=1e-6)
    assert honest["beta"][2] >= 0.0 and honest["ok"]          # граница активна, проекционный градиент нулевой
    assert mutant["beta"][2] < -0.3                           # снятое ограничение: γ < 0 — мутация видна


def test_spline_basis_partition_of_unity_and_five_columns():
    e = np.linspace(-1, 1, 301)
    kn = FB.spline_knots(e)
    B = FB.spline_basis(e, kn)
    assert B.shape == (301, 5) and (B >= -1e-12).all()
    t = np.r_[[kn["lo"]] * 4, kn["inner"], [kn["hi"]] * 4]
    from scipy.interpolate import BSpline
    assert np.allclose(BSpline.design_matrix(e, t, 3).toarray().sum(1), 1.0)
    assert np.allclose(FB.spline_basis(np.array([-9.0, 9.0]), kn), FB.spline_basis(np.array([-1.0, 1.0]), kn))       # вне окна — прижато к границам


def test_c5_gam_recovers_nonlinear_gate_better_than_linear():
    rng = np.random.default_rng(5)
    n = 30000
    ell, lH = rng.uniform(-1, 1, n), rng.normal(0, 0.4, n)
    w = 1 / (1 + np.exp(-3.0 * (ell ** 2 - 0.3)))
    rv = np.exp(lH + w * ell) * rng.gamma(60.0, 1 / 60.0, n)
    tid = np.zeros(n, int)
    kn = FB.spline_knots(ell)
    B = FB.spline_basis(ell, kn)
    mu, sd = CF.standardize(B)
    g = FB.fit_lin(rv, lH, ell, (B - mu) / sd, tid, 1, kappa=1e-6, lam=1e-6)
    lin = FB.fit_lin(rv, lH, ell, ((ell - ell.mean()) / ell.std())[:, None], tid, 1, kappa=1e-6, lam=1e-6)
    wg = 1 / (1 + np.exp(-(g["alpha"] + g["delta"][0] + CF.mv((B - mu) / sd, g["beta"]))))
    wl = 1 / (1 + np.exp(-(lin["alpha"] + lin["delta"][0] + ((ell - ell.mean()) / ell.std()) * lin["beta"][0])))
    assert np.corrcoef(wg, w)[0, 1] > 0.9 > np.corrcoef(wl, w)[0, 1]


def test_c10_eta_zero_reproduces_c1_fit_1e8():
    rng = np.random.default_rng(9)
    rv, lH, ell, Z, _, _ = gate_data(3000, k=2)
    tid = rng.integers(0, 4, len(rv))
    base = FB.fit_lin(rv, lH, ell, Z, tid, 4)
    frozen = FB.fit_lin(rv, lH, ell, Z, tid, 4, eta=True, eta_zero=True)
    assert np.allclose(frozen["eta"], 0.0)
    for key in ("alpha", "delta", "beta"):
        assert np.allclose(frozen[key], base[key], rtol=0, atol=1e-8)
    n, k = 4, Z.shape[1]                                        # целевая функция со свободными H_i при H = 0 совпадает с целевой функцией C-a: значение и градиент по (α, δ, β)
    c, l = FB.scales(ell)
    x = np.r_[rng.normal(0, 0.3, n + 1 + k), np.zeros(n * k)]
    v1, g1 = FB._obj_eta(rv, lH, ell, Z, tid, n, c, l)(x)
    v0, g0 = CF._obj(rv, lH, ell, Z, tid, n, c, l)(x[:n + 1 + k])
    assert abs(v1 - v0) < 1e-12 and np.allclose(g1[:n + 1 + k], g0, rtol=0, atol=1e-12)
    free = FB.fit_lin(rv, lH, ell, Z, tid, 4, eta=True)
    assert np.abs(free["eta"]).max() > 1e-6 and free["ok"]       # мутация «η заморожен всегда» отличима от свободного C10


def test_c8_b_zero_gives_constant_weight_and_known_regime_sign():
    rng = np.random.default_rng(11)
    n = 20000
    ell, lH, z = rng.uniform(-0.8, 0.8, n), rng.normal(0, 0.4, n), rng.normal(0, 1, n)
    pi = 1 / (1 + np.exp(-(-3.0 * z)))                         # высокий уровень рынка -> режим crisis (w = 0,2)
    w = pi * 0.8 + (1 - pi) * 0.2
    rv = np.exp(lH + w * ell) * rng.gamma(60.0, 1 / 60.0, n)
    tid = np.zeros(n, int)
    r = FB.fit_regime(rv, lH, ell, z, tid, 1, kappa=1e-6, lam=1e-6)
    wc = 1 / (1 + np.exp(-(r["a_calm"] + r["d_calm"][0]))); wr = 1 / (1 + np.exp(-(r["a_crisis"] + r["d_crisis"][0])))
    assert r["b"] < -0.5 and wc > wr + 0.2
    z0 = FB.fit_regime(rv, lH, ell, z, tid, 1, b_zero=True)
    assert z0["b"] == 0.0
    st = {**z0, "mu": np.zeros(1), "sd": np.ones(1)}
    F, wt, _ = MD._pred_regime("regime", st, {"X": z[:, None], "tix": tid, "lH": lH, "ell": ell})
    assert np.ptp(wt) < 1e-12                                   # b = 0: вес постоянен


def test_c9_known_three_expert_weights_recovered_and_simplex():
    rng = np.random.default_rng(13)
    n = 20000
    lH = rng.normal(0, 0.5, n)
    lK2, lK3 = lH + rng.normal(0, 0.4, n), lH + rng.normal(0, 0.4, n)
    rv = np.exp(0.2 * lH + 0.5 * lK2 + 0.3 * lK3) * rng.gamma(80.0, 1 / 80.0, n)
    lK = np.stack([lH, lK2, lK3], 1)
    ell = (lK2 + lK3) / 2 - lH
    tr = {"lH": lH, "lK2": lK2, "lK3": lK3}
    Z = MD.three_z(tr)
    mu, sd = CF.standardize(Z)
    r = FB.fit_three(rv, lK, (Z - mu) / sd, np.zeros(n, int), 1, ell, kappa=1e-6, lam=1e-6)
    w3 = FB.softmax(r["alpha"] + r["delta"][0] + np.einsum("ij,jk->ik", (Z - mu) / sd, r["beta"]))
    assert np.allclose(w3.sum(1), 1.0, atol=1e-12) and (w3 >= 0).all()
    assert np.allclose(w3.mean(0), [0.2, 0.5, 0.3], atol=0.08)


def test_c9_mutation_weights_outside_simplex_is_caught(monkeypatch):
    rng = np.random.default_rng(17)
    lH = rng.normal(0, 0.5, 500)
    tr = {"lH": lH, "lK2": lH + 0.1, "lK3": lH - 0.1}
    Z = MD.three_z(tr)
    st = {"alpha": np.zeros(3), "delta": np.zeros((1, 3)), "beta": np.ones((4, 3)) * 0.1, "mu": Z.mean(0), "sd": Z.std(0)}
    pr = {**tr, "tix": np.zeros(500, int)}
    s_ok = MD._pred_three("three", st, pr)[2]
    assert np.allclose(s_ok["wH"] + s_ok["wK2"] + s_ok["wK3"], 1.0, atol=1e-12)
    monkeypatch.setattr(FB, "softmax", lambda e: 1.0 / (1.0 + np.exp(-e)))        # независимые сигмоиды вместо softmax
    s_bad = MD._pred_three("three", st, pr)[2]
    assert not np.allclose(s_bad["wH"] + s_bad["wK2"] + s_bad["wK3"], 1.0, atol=1e-6)
