"""B1-4 (Ж-204): положительность уровневых прогнозов (L1, H1) — PIT-граница ε; тождества H1 / H2-L4."""
import numpy as np

from models_v2.s5 import pit
from models_v2.s5.b1 import designs

N = 50


def spec(kind, **kw):
    z = np.zeros(N)
    return designs.Spec("t", "mse", z[:, None], z, kind, **kw)


def test_negative_level_forecast_is_floored_and_counted():
    eps = np.full(N, 0.07)
    sp = spec("lvl", mult=np.ones(N), add=np.zeros(N))
    f = np.linspace(-3, 3, N)
    s2, n_np, n_fl = designs.sigma2(sp, np.arange(N), f, np.zeros(N), eps)
    assert (s2 > 0).all() and (s2 >= eps).all()
    assert n_np == int((f <= 0).sum()) and n_fl == int((f < 0.07).sum())


def test_mutation_no_floor_gives_nonpositive():
    f = np.linspace(-3, 3, N)
    assert (1.0 * f + 0.0 <= 0).any()   # M-B1-9: без границы сырой прогноз бывает <= 0, тест выше это ловит


def test_eps_floor_uses_only_rows_up_to_d_and_global_min_is_caught():
    rv = np.abs(np.random.default_rng(1).normal(2, 1, 300)) + 0.5
    rv2 = rv.copy()
    rv2[200:] = 0.01
    assert np.array_equal(pit.eps_floor(rv)[:200], pit.eps_floor(rv2)[:200])
    glob = 0.1 * rv2.min() * np.ones(300)                      # M-B1-3: ε по всем данным
    assert not np.array_equal(glob[:200], pit.eps_floor(rv2)[:200])


def test_h1_zero_correction_equals_har_and_h2_l4_zero_equals_har():
    harb = np.exp(np.random.default_rng(2).normal(1, 0.5, N))
    eps = 0.1 * np.ones(N)
    h1s = spec("lvl", mult=harb, add=harb)
    h1r = spec("lvl", mult=np.ones(N), add=harb)
    h2 = spec("lnmu", off=np.log(harb))
    r = np.arange(N)
    assert np.allclose(designs.sigma2(h1s, r, np.zeros(N), np.zeros(N), eps)[0], harb)
    assert np.allclose(designs.sigma2(h1r, r, np.zeros(N), np.zeros(N), eps)[0], harb)
    assert np.allclose(designs.sigma2(h2, r, np.zeros(N), np.zeros(N), eps)[0], harb)
    assert np.allclose(designs.sigma2(h2, r, np.full(N, 0.4), np.zeros(N), eps, c=0.5)[0], harb * np.exp(0.2))


def test_l2_keeps_bias_correction_and_lnmu_does_not():
    off = np.zeros(N)
    a = designs.sigma2(spec("ln", off=off), np.arange(N), np.zeros(N), np.full(N, 0.6), np.zeros(N))[0]
    b = designs.sigma2(spec("lnmu", off=off), np.arange(N), np.zeros(N), np.full(N, 0.6), np.zeros(N))[0]
    assert np.allclose(a, np.exp(0.3)) and np.allclose(b, 1.0)
