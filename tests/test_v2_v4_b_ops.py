"""V4-B, G10: тождества операторов (допуск 1e-10), концы (w = 1 -> H побитно), аналитические производные против численных, известный ответ."""
import numpy as np
import pytest

from models_v2.s11 import v4_b_ops as OP

rng = np.random.default_rng(7)
H = np.exp(rng.normal(0.0, 0.4, 500))
M = H * np.exp(rng.normal(0.0, 0.25, 500))
W = rng.uniform(0.05, 0.95, 500)


def test_p_one_is_b1_and_p_minus_one_is_b3():
    assert np.allclose(OP.forward("B4", H, M, W, p=1.0)[0], OP.forward("B1", H, M, W)[0], rtol=1e-10, atol=0)
    assert np.allclose(OP.forward("B4", H, M, W, p=-1.0)[0], OP.forward("B3", H, M, W)[0], rtol=1e-10, atol=0)


def test_p_to_zero_is_b2():
    b2 = OP.forward("B2", H, M, W)[0]
    assert np.array_equal(OP.forward("B4", H, M, W, p=0.0)[0], b2)
    assert np.allclose(OP.forward("B4", H, M, W, p=1e-10)[0], b2, rtol=1e-10, atol=0)
    assert np.allclose(OP.forward("B4", H, M, W, p=-1e-10)[0], b2, rtol=1e-10, atol=0)


def test_power_formula_continuous_at_taylor_switch():
    lo = OP.forward("B4", H, M, W, p=OP.EPS_P * 0.999)[0]
    hi = OP.forward("B4", H, M, W, p=OP.EPS_P * 1.001)[0]
    assert np.allclose(lo, hi, rtol=1e-8, atol=0)


@pytest.mark.parametrize("cfg", ["B1", "B2", "B3", "B4", "B5"])
def test_w_one_gives_h_bitwise_and_w_zero_gives_m(cfg):
    one, zero = np.ones_like(H), np.zeros_like(H)
    assert np.array_equal(OP.forward(cfg, H, M, one, p=0.7)[0], H)
    assert np.array_equal(OP.forward(cfg, H, M, zero, p=0.7)[0], M)


def test_b5_ends_keep_shift_and_b6_ends():
    one = np.ones_like(H)
    assert np.allclose(OP.forward("B5", H, M, one, b=0.1)[0], H * np.exp(0.1), rtol=1e-14)
    th = np.tile([1.0, 0.0, 0.0], (len(H), 1))
    assert np.array_equal(OP.forward_b6(th, np.log(H), np.log(M), np.log(M)), np.exp(np.log(H)))


@pytest.mark.parametrize("cfg,p,b", [("B1", 0, 0), ("B2", 0, 0), ("B3", 0, 0), ("B4", 0.7, 0), ("B4", -1.3, 0), ("B4", 3e-5, 0), ("B5", 0, 0.07)])
def test_analytic_derivatives_match_numeric(cfg, p, b):
    F, dw, dp, db = OP.forward(cfg, H, M, W, p, b)
    h = 1e-6
    nw = (OP.forward(cfg, H, M, W + h, p, b)[0] - OP.forward(cfg, H, M, W - h, p, b)[0]) / (2 * h)
    assert np.allclose(dw, nw, rtol=1e-5, atol=1e-7)
    if cfg == "B4":
        np_ = (OP.forward(cfg, H, M, W, p + h, b)[0] - OP.forward(cfg, H, M, W, p - h, b)[0]) / (2 * h)
        assert np.allclose(dp, np_, rtol=2e-4, atol=1e-6)
    if cfg == "B5":
        nb = (OP.forward(cfg, H, M, W, p, b + h)[0] - OP.forward(cfg, H, M, W, p, b - h)[0]) / (2 * h)
        assert np.allclose(db, nb, rtol=1e-5, atol=1e-7)


def test_known_answer_b2_equal_models_any_weight():
    """H = M: любой оператор даёт H (известный ответ)."""
    for cfg in ("B1", "B2", "B3", "B4"):
        assert np.allclose(OP.forward(cfg, H, H, W, p=0.5)[0], H, rtol=1e-12)


def test_mutation_wrong_b3_formula_would_be_caught():
    """Арифметическое вместо гармонического (мутация формулы B3) отличается заметно — тождество p = −1 его бы поймало."""
    assert not np.allclose(OP.forward("B1", H, M, W)[0], OP.forward("B4", H, M, W, p=-1.0)[0], rtol=1e-10)
