"""V4-B, G10: оценка параметров — градиенты и гессиан, Ньютон B2 против L-BFGS, известный ответ на синтетике, штраф иерархии (мутация κ = 0), приор."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from models_v2.s11 import v4_b_fit as FT, v4_b_ops as OP, v4_b_solve as SV


def synth(cfg, n_per=(4000, 4000, 4000), w0=0.35, seed=1, abc=(0.5, 0.3, 0.2)):
    """RV = F(H, M; w0) · ξ, E ξ = 1 (ξ — логнормаль со средним 1): QLIKE-оптимум — истинный параметр."""
    rng = np.random.default_rng(seed)
    tid = np.concatenate([np.full(k, i) for i, k in enumerate(n_per)])
    n = len(tid)
    K2 = np.exp(rng.normal(0, 0.5, n))
    K3 = K2 * np.exp(rng.normal(0, 0.2, n))
    H = K2 * np.exp(rng.normal(0, 0.25, n))
    if cfg == "B6":
        F = np.exp(abc[0] * np.log(H) + abc[1] * np.log(K2) + abc[2] * np.log(K3))
    else:
        F = OP.forward(cfg, H, OP.mean_m(K2, K3), np.full(n, w0))[0]
    s = 0.3
    xi = np.exp(rng.normal(-s * s / 2, s, n))
    return FT.make_train(F * xi, H, K2, K3, tid)[0]


@pytest.mark.parametrize("cfg", ["B1", "B2", "B3"])
def test_known_answer_weight_recovered(cfg):
    r = FT.fit(cfg, synth(cfg, w0=0.35))
    # w_p стянут к приору 0,5 с весом 1/(n+1) (n = 3 бумаги, приор и штраф по 250 пар): допуск шире, чем у w_i
    assert r["ok"] and abs(r["w_p"] - 0.35) < 0.06 and np.all(np.abs(r["w"] - 0.35) < 0.05)


def test_known_answer_b6_recovered_on_simplex():
    r = FT.fit("B6", synth("B6"))
    assert r["ok"] and np.allclose(r["theta_p"], [0.5, 0.3, 0.2], atol=0.06) and np.allclose(r["theta"].sum(1), 1.0, atol=1e-8) and (r["theta"] >= -1e-12).all()


def test_newton_b2_equals_lbfgs_optimum():
    T = synth("B2", n_per=(300, 80, 20), w0=0.6)
    a = FT.fit("B2", T)
    x0, lo, hi = FT._layout("B2", T.n)
    x, _, _ = FT._lbfgs("B2", T, FT.curvature(T), FT.KAPPA, FT.LAM, [x0])
    assert a["ok"] and np.allclose(a["_x"], x, atol=1e-5)


def test_b2_gradient_and_hessian_match_numeric():
    T = synth("B2", n_per=(50, 30), w0=0.5)
    fgh = SV.b2_fgh(T, FT.curvature(T), FT.KAPPA, FT.LAM, 0.5)
    x = np.array([0.4, 0.3, 0.7])
    assert check_grad(lambda z: fgh(z)[0], lambda z: fgh(z)[1], x, epsilon=1e-7) < 1e-6
    h = 1e-6
    num = np.array([(fgh(x + h * e)[1] - fgh(x - h * e)[1]) / (2 * h) for e in np.eye(3)])
    assert np.allclose(fgh(x)[2], num, atol=1e-6)


@pytest.mark.parametrize("cfg", ["B1", "B3", "B4", "B5"])
def test_objective_gradient_matches_numeric(cfg):
    T = synth("B2", n_per=(60, 40))
    x0, _, _ = FT._layout(cfg, T.n)
    x = x0 + np.random.default_rng(3).uniform(-0.05, 0.05, len(x0))
    f = FT._obj(cfg, T, FT.curvature(T), FT.KAPPA, FT.LAM)
    assert check_grad(lambda z: f(z)[0], lambda z: f(z)[1], x, epsilon=1e-7) < 1e-6


def test_b6_gradient_matches_numeric():
    T = synth("B6", n_per=(60, 40))
    obj = SV.b6_obj(T, FT.curvature(T), FT.KAPPA, FT.LAM, FT.TH0)
    x = np.random.default_rng(4).uniform(0.1, 0.6, 9)
    assert check_grad(lambda z: obj(z)[0], lambda z: obj(z)[1], x, epsilon=1e-7) < 1e-6


def gap(kappa):
    """Отклонение w_i малой бумаги (20 пар, истинный вес 0,8) от панельного w_p: штраф иерархии стягивает его."""
    rng = np.random.default_rng(11)
    parts = []
    for i, (k, w0) in enumerate(((3000, 0.4), (20, 0.8))):
        K2 = np.exp(rng.normal(0, 0.5, k))
        H = K2 * np.exp(rng.normal(0, 0.3, k))
        F = OP.forward("B2", H, K2, np.full(k, w0))[0]
        parts.append((F * np.exp(rng.normal(-0.045, 0.3, k)), H, K2, K2, np.full(k, i)))
    T = FT.make_train(*[np.concatenate([p[j] for p in parts]) for j in range(5)])[0]
    r = FT.fit("B2", T, kappa=kappa)
    return abs(r["w"][1] - r["w_p"])


def shrinks() -> bool:
    """Боевой вход fit(kappa=None -> KAPPA): штраф иерархии стягивает малую бумагу к панели хотя бы вдвое против κ = 0."""
    return gap(None) < 0.5 * gap(0.0) and gap(0.0) > 0.2


def test_hierarchy_penalty_shrinks_small_ticker():
    assert shrinks()


def test_mutation_kappa_zero_is_caught(monkeypatch):
    """κ = 0 (δ_i без штрафа): стягивания нет, результат меняется — проверка стягивания падает."""
    monkeypatch.setattr(FT, "KAPPA", 0.0)
    assert not shrinks()


def test_prior_below_min_pairs_is_neutral():
    pr = FT.prior_params("B6")
    assert FT.MIN_PAIRS == 60 and np.allclose(pr["theta_p"], 1 / 3) and FT.prior_params("B2")["w_p"] == 0.5 and FT.prior_params("B4")["p"] == 0.0 and FT.prior_params("B5")["b_p"] == 0.0
