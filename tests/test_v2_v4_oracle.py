"""G7/G8 (г): oracle bound на синтетике с известным ответом и свойства порядка; мутация «oracle без clip» ловится."""
import numpy as np
import pytest

from models_v2.s11 import v4_oracle as OR


def ql(r, f):
    return r / f - np.log(r / f) - 1.0


def test_known_answer_o2_is_clip_and_qlike_exact():
    rv, h, m = np.array([2.0, 5.0, 0.5, 3.0]), np.ones(4), 3 * np.ones(4)
    x2 = OR.o2_series(rv, h, m)
    assert x2.tolist() == [2.0, 3.0, 1.0, 3.0]
    assert OR.q(rv, x2) == pytest.approx((ql(5.0, 3.0) + ql(0.5, 1.0)) / 4, abs=1e-14)
    assert OR.known_answer_ok()


def test_mutation_oracle_without_clip_is_caught(monkeypatch):
    """Без clip F = RV: QLIKE = 0 и на днях вне диапазона — известный ответ нарушен."""
    monkeypatch.setattr(OR, "_clip", lambda x, lo, hi: x)
    assert not OR.known_answer_ok()
    rv, h, m = np.array([5.0, 0.5]), np.ones(2), 3 * np.ones(2)
    got = OR.q(rv, OR.o2_series(rv, h, m))
    assert got == 0.0 and got != pytest.approx((ql(5.0, 3.0) + ql(0.5, 1.0)) / 2)


def test_inside_interval_loss_is_zero():
    rv = np.array([1.5, 2.0, 2.5])
    assert OR.q(rv, OR.o2_series(rv, np.ones(3), 3 * np.ones(3))) == 0.0


def test_order_and_nonnegative_and_parts_sum_on_random_panels():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(40, 300))
        rv = np.exp(rng.normal(0, 0.7, n))
        h, m, t = (rv * np.exp(rng.normal(s, 0.3, n)) for s in (0.1, -0.05, 0.0))
        o = OR.oracle(rv, h, m, third=t)
        e = 1e-12
        assert min(o[k] for k in ("q_h", "q_m", "o1_arith", "o1_geom", "o1", "o2", "o3", "o4")) >= 0
        assert o["o2"] <= o["o1"] + e and o["o1"] <= o["q_best"] + e and o["o3"] <= o["o2"] + e
        assert o["o4"] <= OR.q(rv, OR.o2_series(rv, h, m)) + e
        assert o["p1_const_weight"] + o["p2_conditional"] + o["p3_out_of_range"] + o["p4_residual"] == pytest.approx(o["q_best"], abs=1e-12)
        assert sum(o["s" + k] for k in ("1_const_weight", "2_conditional", "3_out_of_range", "4_residual")) == pytest.approx(1.0, abs=1e-12)


def test_best_weight_recovers_known_weight():
    rng = np.random.default_rng(3)
    h, m = np.exp(rng.normal(0, 0.5, 400)), np.exp(rng.normal(0.3, 0.5, 400))
    w_a, q_a = OR.best_weight(0.7 * h + 0.3 * m, h, m, False)
    w_g, q_g = OR.best_weight(h ** 0.6 * m ** 0.4, h, m, True)
    assert w_a == pytest.approx(0.3, abs=1e-4) and q_a < 1e-9
    assert w_g == pytest.approx(0.4, abs=1e-4) and q_g < 1e-9


def test_bias_multiplier_known_answer():
    x = np.array([1.0, 2.0, 3.0])
    f, b = OR.o3_series(1.25 * x, x)
    assert b == pytest.approx(1.25) and OR.q(1.25 * x, f) == pytest.approx(0.0, abs=1e-14)


def test_o3_multiplier_is_qlike_optimal():
    rng = np.random.default_rng(5)
    x = np.exp(rng.normal(0, 0.4, 200))
    rv = x * np.exp(rng.normal(0.1, 0.5, 200))
    f, b = OR.o3_series(rv, x)
    assert all(OR.q(rv, c * x) >= OR.q(rv, f) - 1e-12 for c in (0.8 * b, 0.95 * b, 1.05 * b, 1.3 * b))


def test_empty_or_nonpositive_input_refused():
    one = np.ones(3)
    with pytest.raises(ValueError):
        OR.oracle(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError):
        OR.oracle(np.array([1.0, 0.0, 1.0]), one, one)
    with pytest.raises(ValueError):
        OR.oracle(one, one, np.array([1.0, np.nan, 1.0]))


def test_swapped_sign_of_decomposition_is_caught():
    """Перепутанный порядок частей (O2 ↔ O3) нарушает суммирование/неотрицательность частей."""
    rng = np.random.default_rng(11)
    rv = np.exp(rng.normal(0, 0.6, 200))
    h, m = rv * np.exp(rng.normal(0.2, 0.3, 200)), rv * np.exp(rng.normal(-0.1, 0.3, 200))
    o = OR.oracle(rv, h, m)
    assert o["p1_const_weight"] >= -1e-12 and o["p2_conditional"] >= -1e-12 and o["p3_out_of_range"] >= -1e-12 and o["p4_residual"] >= 0
