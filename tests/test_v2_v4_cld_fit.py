"""V4-CLD (G32): регрессии D — известные ответы (гребень, Хьюбер, сплайн) и отображения в вес; тождество D̂ ≡ 0 → w = 0,5."""
import numpy as np
import pytest

from models_v2.s11 import v4_cld_fit as CF


def data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 3))
    return Z, rng


def test_ridge_known_answer_lam_zero_and_closed_form_lam_250():
    Z, rng = data()
    y = 2.0 + 3.0 * Z[:, 0] + rng.normal(0, 0.01, len(Z))
    a, b = CF.ridge_w(Z, y, lam=0.0)
    assert a == pytest.approx(2.0, abs=1e-2) and b[0] == pytest.approx(3.0, abs=1e-2)
    Zc, yc = Z - Z.mean(0), y - y.mean()
    a2, b2 = CF.ridge_w(Z, y)                                            # λ = 250 пар-эквивалентов
    exp = np.linalg.solve(Zc.T @ Zc + 250.0 * np.eye(3), Zc.T @ yc)
    assert np.allclose(b2, exp) and a2 == pytest.approx(y.mean() - Z.mean(0) @ exp)
    assert abs(b2[0]) < abs(b[0])                                        # гребень сжимает


def test_ridge_sign_not_swapped():
    Z, rng = data()
    y = -1.5 * Z[:, 1] + rng.normal(0, 0.1, len(Z))
    r = CF.fit_model("CLD1", Z, y)
    assert r["coef"][1] < -1.0 and np.sign(r["predict"](np.array([[0.0, 1.0, 0.0]]))[0]) == -1


def test_huber_resists_heavy_tails_better_than_ridge():
    Z, rng = data()
    y = 1.0 * Z[:, 0] + rng.standard_t(1.5, len(Z)) * 0.5
    r1, r3 = CF.fit_model("CLD1", Z, y), CF.fit_model("CLD3", Z, y)
    assert abs(r3["coef"][0] - 1.0) < abs(r1["coef"][0] - 1.0) and r3["ok"]


def test_huber_equals_ridge_when_delta_huge():
    Z, rng = data(800)
    y = Z[:, 0] + rng.normal(0, 1, 800)
    a, b = CF.ridge_w(Z, y)
    a2, b2, _, ok = CF.huber(Z, y, delta=1e9)
    assert ok and np.allclose(b, b2) and a == pytest.approx(a2)


def test_spline_catches_nonlinearity_ridge_cannot():
    Z, rng = data()
    y = Z[:, 0] ** 2 + rng.normal(0, 0.1, len(Z))
    zq = np.column_stack([np.linspace(-2, 2, 9), np.zeros(9), np.zeros(9)])
    p4, p1 = CF.fit_model("CLD4", Z, y)["predict"](zq), CF.fit_model("CLD1", Z, y)["predict"](zq)
    assert np.abs(p4 - (zq[:, 0] ** 2 - 1.0 + (y.mean() - (-1.0 + 1.0)))).max() < 0.6 and np.ptp(p1) < 0.5 < np.ptp(p4)
    assert CF.fit_model("CLD4", Z, y)["coef"][0] > 5 * CF.fit_model("CLD4", Z, y)["coef"][1]


def test_spline_has_four_df_per_feature_and_clips_outside_window():
    Z, _ = data(500)
    sp = CF.Spline().fit(Z)
    assert sp.transform(Z).shape == (500, 12)
    assert np.allclose(sp.transform(np.full((2, 3), 50.0)), sp.transform(np.full((2, 3), 1e6)))      # вне окна — значение на границе
    assert np.allclose(sp.transform(np.full((1, 3), 50.0)), sp.transform(sp.hi[None, :]))


def test_weights_identity_dhat_zero_gives_half_for_every_config():
    for cfg in CF.CFGS:
        assert np.array_equal(CF.weight(cfg, np.zeros(5), 0.7, 1.3), np.full(5, 0.5))


def test_weight_mappings_known_answers_and_bounds():
    d = np.array([-100.0, -0.5, 0.0, 0.5, 100.0])
    w1 = CF.weight("CLD1", d, 0.5, 9.0)
    assert w1[3] == pytest.approx(1 / (1 + np.exp(-1.0))) and w1[0] < 1e-6 and w1[4] > 1 - 1e-6
    w2 = CF.weight("CLD2", d, 9.0, 2.0)
    assert np.allclose(w2, [0.0, 0.375, 0.5, 0.625, 1.0])
    assert CF.weight("CLD2", d, 9.0, 0.0).tolist() == [0.5] * 5


def test_unknown_config_rejected():
    with pytest.raises(ValueError):
        CF.fit_model("CLD9", np.zeros((10, 2)), np.zeros(10))
