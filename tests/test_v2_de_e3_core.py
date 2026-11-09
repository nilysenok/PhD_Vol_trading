"""E3 (X3): ядро — g_a = 0 ⇒ F = A0g побитно, градиент гребня QLIKE, известный ответ, a_t ∈ [0; 1] и его поведение на полезной / вредной g_a (синтетика)."""
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from models_v2.s15 import de_e3_core as C

NZ = 9


def synth(n=700, seed=3, signal=0.5, noise=0.3):
    d = pd.bdate_range("2016-12-30", periods=n)
    rng = np.random.default_rng(seed)
    tp = d[1:].append(pd.DatetimeIndex([d[-1] + pd.offsets.BDay(1)])).to_numpy("datetime64[ns]")
    F0 = np.exp(0.2 * rng.standard_normal(n))
    Z = rng.standard_normal((n, NZ))
    rv = F0 * np.exp(signal * Z[:, 0] + noise * rng.standard_normal(n))
    return d, tp, F0, rv, Z


def test_ga_zero_gives_A0g_bitwise_and_a_zero_too():
    d, tp, F0, rv, Z = synth()
    assert np.array_equal(C.forecast(F0, np.zeros(len(F0)), np.full(len(F0), 0.7)), F0)
    assert np.array_equal(C.forecast(F0, np.full(len(F0), 0.3), np.zeros(len(F0))), F0)
    w = C.walk_forward_ga(F0[:150], rv[:150], tp[:150], d[:150], Z[:150])           # меньше MIN_TRAIN закрытых пар — модели нет
    assert not w["fitted"].any() and np.array_equal(C.forecast(F0[:150], w["ga"], np.ones(150)), F0[:150])


def test_gradient_equals_numeric():
    d, tp, F0, rv, Z = synth(n=120)
    R = rv / F0
    th = 0.1 * np.random.default_rng(1).standard_normal(NZ + 1)
    for lam in C.LAMS:
        g = C.loss_grad(th, Z, R, lam)[1]
        num = approx_fprime(th, lambda t: C.loss_grad(t, Z, R, lam)[0], 1e-6)
        assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_known_answer_ridge_recovers_signal_and_shrinks_with_lambda():
    d, tp, F0, rv, Z = synth(n=400, noise=0.0)
    R = rv / F0
    mu, sd = C.norm_stats(Z)
    Xs = (Z - mu) / sd
    th_small, th_big = C.ridge_fit(Xs, R, 1e-6), C.ridge_fit(Xs, R, 10.0)
    assert np.allclose(C.apply(th_small, Xs), 0.5 * Z[:, 0], atol=1e-3)
    assert np.linalg.norm(th_big[1:]) < np.linalg.norm(th_small[1:])                    # гребень сжимает коэффициенты при Z


def _a(signal_sign):
    d, tp, F0, rv, Z = synth(signal=0.5)
    w = C.walk_forward_ga(F0, rv, tp, d, Z)
    ga = signal_sign * w["ga"]
    return C.adaptive_weights(rv, F0, ga, w["fitted"], tp, d), w


def test_a_in_unit_interval_zero_before_history_high_if_helpful_low_if_harmful():
    a_good, w = _a(+1.0)
    a_bad, _ = _a(-1.0)
    assert ((a_good >= 0) & (a_good <= 1)).all() and ((a_bad >= 0) & (a_bad <= 1)).all()
    assert (a_good[:200] == 0).all() and w["fitted"].any()                           # меньше MIN_DELTA пар с g_a — a = 0
    assert a_good[-100:].mean() > 0.9 and a_bad[-100:].mean() < 0.01


def test_cusum_state_known_answer():
    S, s = C.cusum_state(np.array([1.0, -3.0, 2.0, 2.0]))
    assert S == 4.0 and abs(s - np.std([1.0, -3.0, 2.0, 2.0])) < 1e-15                # max(0, 1) → 0 → 2 → 4
