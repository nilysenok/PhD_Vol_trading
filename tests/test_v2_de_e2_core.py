"""E2 (X2): модель хвостовой смеси — градиент, тождество c_n = c_s = 1, диапазоны, известный ответ, признаки и метка (синтетика, numpy)."""
import numpy as np
import pandas as pd

from models_v2.s15 import de_e2_feat as FE, de_e2_model as M


def _data(n=80, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, FE.NP)), np.exp(0.3 * rng.standard_normal(n)), (rng.random(n) < 0.2).astype(float)


def test_gradient_equals_numeric_for_both_betas():
    X, R, y = _data()
    th = 0.3 * np.random.default_rng(2).standard_normal(M.NTH)
    for beta in M.BETAS:
        L, g = M.loss_grad(th, X, R, y, beta)
        num = np.empty_like(th)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = 1e-6
            num[i] = (M.loss_grad(th + e, X, R, y, beta)[0] - M.loss_grad(th - e, X, R, y, beta)[0]) / 2e-6
        assert np.max(np.abs(g - num)) < 1e-7, beta


def test_beta_term_is_in_loss_and_gradient():
    X, R, y = _data()
    th = 0.3 * np.random.default_rng(3).standard_normal(M.NTH)
    (l1, g1), (l2, g2) = M.loss_grad(th, X, R, y, 0.1), M.loss_grad(th, X, R, y, 1.0)
    assert l1 != l2 and not np.allclose(g1[2 * M.NC:], g2[2 * M.NC:]) and np.allclose(g1[:2 * M.NC], g2[:2 * M.NC])    # β меняет только градиент p-части при фиксированном p


def test_cn_cs_one_gives_F_equals_H_bitwise_for_any_p():
    X, R, _ = _data()
    th = np.zeros(M.NTH)
    th[2 * M.NC:] = np.random.default_rng(4).standard_normal(M.NP + 1) * 3            # p произвольно, c_n = c_s = 1
    f = M.forward(th, X)
    assert np.array_equal(f["cn"], np.ones(len(X))) and np.array_equal(f["cs"], np.ones(len(X))) and np.array_equal(f["m"], np.ones(len(X)))
    H = np.exp(np.random.default_rng(5).standard_normal(len(X)))
    assert np.array_equal(H * np.exp(M.predict([th], X)["lnm"]), H)


def test_mixture_equals_formula_p_in_unit_interval_and_F_positive():
    X = 5 * np.random.default_rng(6).standard_normal((200, FE.NP))
    th = 2 * np.random.default_rng(7).standard_normal(M.NTH)
    f = M.forward(th, X)
    assert np.allclose(f["m"], (1 - f["p"]) * f["cn"] + f["p"] * f["cs"], rtol=1e-14, atol=0)
    assert (f["p"] >= 0).all() and (f["p"] <= 1).all() and (f["m"] > 0).all()
    assert (f["cn"] <= np.exp(M.CMAX)).all() and (f["cn"] >= np.exp(-M.CMAX)).all()            # c = exp(c_max·tanh) ∈ [e⁻¹, e]


def test_known_answer_stress_scale_is_recovered():
    rng = np.random.default_rng(8)
    n = 1500
    X = rng.standard_normal((n, FE.NP))
    y = (X[:, 9] > 0.8).astype(float)                                              # метка определяется колонкой e_d (признак)
    R = (1.0 + 1.2 * y) * rng.gamma(40, 1 / 40, n)                                  # при стрессе RV/H ≈ 2,2
    pr = M.predict([M.fit(X, R, y, 1.0, 0)], X)
    assert pr["p"][y == 1].mean() > 0.6 > 0.25 > pr["p"][y == 0].mean()
    assert pr["cs"].mean() > pr["cn"].mean() + 0.5


def test_features_semivariance_known_answer_and_no_label_column():
    days = pd.bdate_range("2020-01-01", periods=3)
    r = np.array([[-1.0, 2.0, np.nan], [1.0, 1.0, 1.0], [np.nan, np.nan, np.nan]])
    out = FE.rs_neg_share({"days": days, "r": r}, days)
    assert np.allclose(out[:2], [1 / 5, 0.0]) and np.isnan(out[2])
    X = FE.day_inputs(np.zeros((3, FE.NZ)), np.array([0, 1, 0]), out)
    assert X.shape == (3, FE.NP) and len(FE.FEATURES) == FE.NP == 11 and FE.FEATURES[-2:] == ["e_d", "rs_neg_share"]


def test_stress_threshold_uses_only_closed_pairs():
    rv = np.arange(1.0, 101.0)
    closed = np.arange(50)
    assert FE.stress_threshold(rv, closed) == np.quantile(rv[:50], 0.9)
    rv2 = rv.copy()
    rv2[50:] = 1e6                                                                  # будущие пары не меняют порог
    assert FE.stress_threshold(rv2, closed) == FE.stress_threshold(rv, closed)
    assert FE.stress_threshold(rv2, np.arange(100)) != FE.stress_threshold(rv, closed)    # мутация «все пары» меняет
