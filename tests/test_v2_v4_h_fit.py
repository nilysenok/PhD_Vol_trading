"""V4-H (G39): оценка поправки g = θᵀz̃ — градиент, известные ответы, ортогонализация только по окну оценки, гребень, обрезка. Данные проекта не читаются."""
from __future__ import annotations

import numpy as np
from scipy.optimize import check_grad

from models_v2.s11 import v4_h_fit as HF


def sample(n=1600, seed=0, beta=(0.3, 0.0), nt=4, sig=0.2):
    """z̃ (стандартные независимые), x = RV/K8 = exp(β·z̃ + шум) / E: известный θ."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, len(beta)))
    tid = rng.integers(0, nt, n)
    x = np.exp(np.einsum("ij,j->i", Z, np.array(beta)) + rng.normal(0, sig, n) - sig ** 2 / 2)
    return x, Z, tid, nt


def test_gradient_matches_finite_differences_flat_and_hier():
    x, Z, tid, nt = sample(300, 1)
    for hier in (False, True):
        k = Z.shape[1]
        size = k + (nt * k if hier else 0)
        f = HF._obj(x, Z, tid, nt, hier, 250.0, 250.0)
        p0 = np.random.default_rng(2).normal(0, 0.1, size)
        assert check_grad(lambda p: f(p)[0], lambda p: f(p)[1], p0) < 1e-6


def test_known_answer_sign_and_order_of_theta():
    x, Z, tid, nt = sample(3200, 3, beta=(0.3, -0.2, 0.0))
    th = HF.fit(x, Z, tid, nt)["theta"]
    assert 0.2 < th[0] < 0.31 and -0.21 < th[1] < -0.12 and abs(th[2]) < 0.04          # ridge κ = 250 при N = 3200 сжимает слабо


def test_ridge_shrinks_toward_zero():
    x, Z, tid, nt = sample(1600, 4)
    a, b = HF.fit(x, Z, tid, nt, kappa=250.0)["theta"][0], HF.fit(x, Z, tid, nt, kappa=1e6)["theta"][0]
    assert 0 < abs(b) < 0.02 < abs(a)


def test_converged_flag_and_zero_theta_when_no_signal():
    x, Z, tid, nt = sample(1600, 5, beta=(0.0, 0.0))
    r = HF.fit(x, Z, tid, nt)
    assert r["ok"] and np.abs(r["theta"]).max() < 0.05


def test_hierarchical_eta_recovers_ticker_specific_slopes():
    rng = np.random.default_rng(6)
    n, nt = 4000, 2
    Z, tid = rng.normal(0, 1, (n, 1)), rng.integers(0, nt, n)
    b = np.where(tid == 0, 0.35, -0.05)
    x = np.exp(b * Z[:, 0] + rng.normal(0, 0.2, n))
    r = HF.fit(x, Z, tid, nt, hier=True)
    assert r["ok"] and r["eta"][0, 0] > 0.1 and r["eta"][1, 0] < -0.1 and abs((r["eta"][0, 0] + r["theta"][0]) - (r["eta"][1, 0] + r["theta"][0]) - 0.4) < 0.1      # разность наклонов бумаг ≈ 0,40 (гребень сжимает слабо)


def test_orthogonalization_removes_base_and_keeps_new_part():
    rng = np.random.default_rng(7)
    n = 3000
    B = rng.normal(0, 1, (n, 4))
    u = rng.normal(0, 1, n)
    z = (2.0 * B[:, 0] - 1.0 * B[:, 2] + u)[:, None]
    p = HF.orthogonalize_fit(z, B)
    zt = HF.orthogonalize_apply(p, z, B)[:, 0]
    raw = abs(np.corrcoef(z[:, 0], B[:, 0])[0, 1])               # ≈ 0,89; гребень λ = 250 при N = 3000 оставляет ≈ 8 % коэффициента проекции
    assert np.abs(np.corrcoef(zt, B[:, 0])[0, 1]) < 0.2 < raw and np.abs(np.corrcoef(zt, B[:, 2])[0, 1]) < 0.2 and np.corrcoef(zt, u)[0, 1] > 0.95
    assert abs(zt.mean()) < 1e-9 and abs(zt.std() - 1) < 1e-9


def test_mutation_no_projection_leaves_base_in_feature_is_caught():
    """Мутация «без ортогонализации» (гребень проекции огромный): остаток коррелирует с базой — проверка корреляции ловит."""
    rng = np.random.default_rng(8)
    B = rng.normal(0, 1, (3000, 4))
    z = (2.0 * B[:, 0] + rng.normal(0, 1, 3000))[:, None]
    zt = HF.orthogonalize_apply(p := HF.orthogonalize_fit(z, B, lam=1e12), z, B)[:, 0]
    assert abs(np.corrcoef(zt, B[:, 0])[0, 1]) > 0.5 and p is not None


def test_nonfinite_feature_gives_zero_and_clip_bounds():
    z, B = np.array([[1.0], [np.nan], [-1.0], [0.5]]), np.random.default_rng(9).normal(0, 1, (4, 4))
    p = HF.orthogonalize_fit(z[[0, 2, 3]], B[[0, 2, 3]])
    out = HF.orthogonalize_apply(p, z, B)
    assert out[1, 0] == 0.0 and np.isfinite(out).all()
    assert np.array_equal(HF.clip_g(np.array([-3.0, -0.2, 0.7])), np.array([-0.5, -0.2, 0.5]))
