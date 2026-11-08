"""V4-E, G23: E5 (Калман-фильтр) и E6 (совместная (b, w)): рекурсия на известном ответе, фильтр вместо сглаживания, PIT, восстановление параметров."""
from __future__ import annotations

from unittest import mock

import numpy as np

from models_v2.s11 import v4_e_calib as CAL, v4_e_e6 as E6, v4_e_k5 as K5, v4_e_kalman as KF


def test_filter_recursion_known_answer():
    m, P, q, lf = KF.filt(np.array([1.0, 2.0, 4.0]), 0.5)
    pr = 1.5
    k = pr / (pr + 1)
    m1 = 1.0 + k * 1.0
    p1 = pr / (pr + 1)
    k2 = (p1 + 0.5) / (p1 + 1.5)
    assert np.isclose(m[1], m1) and np.isclose(P[1], p1) and np.isclose(m[2], m1 + k2 * (4.0 - m1))


def test_constant_level_recovers_mean_plus_half_variance():
    rng = np.random.default_rng(3)
    mu, sd = np.log(1.2), 0.5
    u = rng.normal(mu, sd, 3000)
    rho, rhat, _ = KF.params([u, u + 0.0], [True, True])
    assert rho[0] < 0.01                                   # уровень постоянен — q/r мало
    b = KF.b_from_state(*KF.filt(u, rho[0])[:2], rho[0], rhat[0])[-1]
    assert abs(b - (mu + sd ** 2 / 2)) < 0.06


def _panel(n=400, nT=3, seed=1):
    rng = np.random.default_rng(seed)
    date = np.datetime64("2018-01-01") + np.repeat(np.arange(n), nT).astype("timedelta64[D]")
    tix = np.tile(np.arange(nT), n)
    F0 = np.exp(rng.normal(0, 0.3, n * nT))
    rv = F0 * np.exp(rng.normal(-0.1, 0.5, n * nT))
    return rv, F0, tix, date, date + np.timedelta64(1, "D")


def test_e5_pit_and_smoother_mutation_caught():
    rv, F0, tix, date, tp = _panel()
    r = 330 * 3
    rv2 = rv.copy()
    rv2[r:] *= 8.0
    run = lambda x: K5.forecast_b(x, F0, tix, date, tp, 3, np.ones(3, bool))[0]
    a, b = run(rv), run(rv2)
    assert np.array_equal(a[:r + 3], b[:r + 3]) and (a[:60 * 3] == 0).all()

    orig = KF.filt

    def smoother(u, rho):                                  # сглаживание: уровень берётся по данным на 20 наблюдений вперёд
        m, P, q, lf = orig(u, rho)
        return np.r_[m[20:], np.repeat(m[-1], 20)], P, q, lf
    with mock.patch.object(KF, "filt", smoother):
        a2, b2 = run(rv), run(rv2)
    assert not np.array_equal(a2[:r + 3], b2[:r + 3])


def test_e5_closed_plus_one_mutation_caught():
    rv, F0, tix, date, tp = _panel()
    r = 330 * 3
    rv2 = rv.copy()
    rv2[r:] *= 8.0
    with mock.patch.object(CAL, "closed_date", lambda t, d: d):
        a, b = (K5.forecast_b(x, F0, tix, date, tp, 3, np.ones(3, bool))[0] for x in (rv, rv2))
    assert not np.array_equal(a[:r + 3], b[:r + 3])


def test_e6_recovers_parameters_and_excluded_gets_panel():
    rng = np.random.default_rng(5)
    n, nT, b0, w0 = 1500, 3, 0.1, 0.4
    H = np.exp(rng.normal(0, 0.3, n * nT))
    M = H * np.exp(rng.normal(0, 0.4, n * nT))
    tix = np.tile(np.arange(nT), n)
    F = H * np.exp(b0 + w0 * np.log(M / H))
    rv = F * rng.exponential(1.0, n * nT)                  # E[RV/F] = 1 -> QLIKE-оптимум в F
    f = E6.fit(rv, H, M, tix, nT)
    assert f["ok"] and abs(f["w_p"] - w0) < 0.12 and abs(f["b_p"] - b0) < 0.12
    date = np.datetime64("2015-01-01") + np.repeat(np.arange(n), nT).astype("timedelta64[D]")
    _, w, b, recs = E6.monthly_base(rv, H, M, M, tix, date, date + np.timedelta64(1, "D"), nT, exclude=2)       # K2 = K3 = M -> √(K2·K3) = M
    last = date >= np.datetime64(date[-1].astype("datetime64[M]"), "D")
    assert np.allclose(b[last & (tix == 2)], recs[-1]["b_p"]) and np.isfinite(w).all()           # исключённая — панельные параметры (δ = 0)
