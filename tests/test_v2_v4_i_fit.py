"""V4-I (G42): оценки компонент — известные ответы на синтетике (наклон HAR-C, вероятность скачка, стягивание m̂, θ ∈ [0, 1]). Данные проекта не читаются."""
from __future__ import annotations

import numpy as np

from models_v2.s11 import v4_i_fit as FT


def test_har_recovers_slope_and_lognormal_correction():
    rng = np.random.default_rng(0)
    n, nT = 60000, 3
    tid = rng.integers(0, nT, n)
    X = rng.normal(size=(n, 3))
    y = 0.4 + 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.3, n)
    p = FT.har_fit(X, y, tid, nT)
    assert abs(p["theta"][1] / p["sd"][0] - 0.5) < 0.02 and abs(p["theta"][2] / p["sd"][1] + 0.2) < 0.02
    assert abs(p["sig2"] - 0.09) < 0.01
    f = FT.har_predict(p, p["mu"][None, :], np.array([-1]))                       # в центре признаков: exp(b0 + σ̂²/2)
    assert abs(np.log(f[0]) - (0.4 + 0.5 * 0.09)) < 0.03


def test_har_unseen_ticker_has_no_own_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    p = FT.har_fit(X, rng.normal(size=500), rng.integers(0, 2, 500), 2)
    p["theta"][4] = 0.5                                                           # свободный член бумаги 0 задан явно: бумага −1 его не получает
    a, b = FT.har_predict(p, X[:3], np.array([-1, -1, -1])), FT.har_predict(p, X[:3], np.array([0, 0, 0]))
    assert np.allclose(b / a, np.exp(0.5))


def test_logit_recovers_probability_and_converges():
    rng = np.random.default_rng(2)
    n = 40000
    X = rng.normal(size=(n, 3))
    pr = 1.0 / (1.0 + np.exp(-(0.3 + 1.0 * X[:, 0])))
    y = (rng.uniform(size=n) < pr).astype(float)
    tid = rng.integers(0, 2, n)
    p = FT.logit_fit(X, y, tid, 2)
    ph = FT.logit_predict(p, X, tid)
    assert p["ok"] and np.abs(ph - pr).mean() < 0.02 and np.corrcoef(ph, X[:, 0])[0, 1] > 0.9


def test_shrink_m_known_answers():
    assert np.isclose(FT.shrink_m([5.0], [0], 1.0)[0], 1.0)                      # нет событий — панельная
    assert np.isclose(FT.shrink_m([5.0], [250.0], 1.0)[0], 3.0)                  # n = κ — середина
    assert abs(FT.shrink_m([5.0], [1e9], 1.0)[0] - 5.0) < 1e-6
    assert np.isclose(FT.shrink_m([np.nan], [10.0], 2.0)[0], 2.0)


def test_theta_known_answer_and_bounds():
    rng = np.random.default_rng(3)
    u = rng.normal(0, 0.3, 400000)
    x = np.exp(0.6 * u)                                                          # RV/K8 = exp(0,6·u) -> оптимум θ = 0,6
    assert abs(FT.theta_fit(x, u) - 0.6) < 0.02
    assert FT.theta_fit(np.exp(2.0 * u), u) > 0.999                              # θ ≤ 1
    assert FT.theta_fit(np.exp(-0.5 * u), u) < 1e-3                              # θ ≥ 0: знак не переворачивается
