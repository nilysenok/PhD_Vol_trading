"""S-L-IMPL-1 (Ж-256 п. 5; `S_L_SPEC.md` разд. 2, 3, 6): модель `l_model` — известный ответ, граница, штраф, три вида π обучения, c и m уровня обучения (В-3, В-4)."""
import numpy as np
import pytest
import torch

from models_v2.s18 import l_model as ML
from models_v2.s18 import r_loss as RL


def arr(n=400, a_true=0.3, pi=1.0, seed=0):
    """RV/cfg8 = e^{a_true}·(1 ± 0,4) — среднее e^{a_true} точно; π ≡ pi, ln B = 0, β·g = 0, cfg8 = 1."""
    z = np.where(np.arange(n) % 2 == 0, 1.4, 0.6)
    pis = np.full(n, pi) if np.isscalar(pi) else pi
    return dict(rv=np.exp(a_true) * z, lnb=np.zeros(n), pi=pis, bg=np.zeros(n), lnc8=np.zeros(n))


def fit(A, lam=0.0, c=1e6, m=1.0, kind="pi", T=None, start=0.0):
    T = np.arange(len(A["rv"])) if T is None else T
    return ML.fit_kind(kind, A, T, lam, c, m, start)


def test_known_answer_pi_one_kappa_huge_a_equals_ln_mean_rv_over_cfg8():
    A = arr()
    assert fit(A) == pytest.approx(np.log(np.mean(A["rv"])), abs=1e-9) == pytest.approx(0.3, abs=1e-9)


def test_bound_clips_optimum_08_to_05_and_minus_08_to_minus_05():
    assert fit(arr(a_true=0.8)) == pytest.approx(0.5, abs=1e-12)
    assert fit(arr(a_true=-0.8)) == pytest.approx(-0.5, abs=1e-12)
    assert ML.A_BOUND == 0.5


def test_penalty_lambda_a_squared_shrinks_to_zero():
    A = arr()
    a0, a1, a2 = fit(A, 0.0), fit(A, 1e-3), fit(A, 1e-1)
    assert abs(a2) < abs(a1) < abs(a0)
    g = lambda a, lam: float(ML.loss_l(torch.tensor(a, dtype=torch.float64), *[torch.as_tensor(A[k]) for k in ("rv", "lnb", "pi", "bg", "lnc8")], lam, 1e6, 1.0))
    assert g(0.2, 0.1) - g(0.2, 0.0) == pytest.approx(0.1 * 0.04, abs=1e-15)


def test_starts_do_not_matter_convex():
    A = arr(pi=np.random.default_rng(1).uniform(0.2, 0.9, 400))
    r = [fit(A, 1e-2, start=s) for s in (-0.3, 0.0, 0.3)]
    assert max(r) - min(r) < 1e-8


def test_train_pi_three_kinds():
    pi = np.random.default_rng(2).uniform(size=2198)
    T = np.arange(100, 900)
    assert np.array_equal(ML.train_pi("pi", pi, T), pi[T])
    pb = ML.train_pi("pibar", pi, T)
    assert (pb == pi[T].mean()).all() and len(pb) == len(T)
    ps = ML.train_pi("shift252", pi, np.arange(2198))
    assert np.array_equal(ps, pi[(np.arange(2198) - 252) % 2198]) and ps[0] == pi[2198 - 252] and ps[251] == pi[2197] and ps[252] == pi[0]
    with pytest.raises(ValueError):
        ML.train_pi("shift1", pi, T)


def test_stats_use_level_of_training_not_w5g_for_controls():
    rng = np.random.default_rng(3)
    n = 800
    A = dict(rv=np.exp(rng.normal(size=n) * 0.4), lnb=rng.normal(size=n) * 0.2, pi=rng.uniform(0.1, 0.9, n), bg=np.zeros(n), lnc8=rng.normal(size=n) * 0.1)
    T = np.arange(50, 700)
    for kind in ("pi", "pibar", "shift252"):
        pt = ML.train_pi(kind, A["pi"], T)
        x = A["rv"][T] / np.exp((1 - pt) * A["lnb"][T] + pt * A["lnc8"][T] + A["bg"][T])
        assert ML.stats_l(kind, A, T) == pytest.approx(RL.clip_stats(x, np.arange(len(T))), rel=1e-13)
    assert ML.stats_l("pibar", A, T) != ML.stats_l("pi", A, T) != ML.stats_l("shift252", A, T) and ML.stats_l("pi", A, T)[1] != 1.0


def test_control_a_score_at_zero_is_zero_and_fit_is_exactly_zero():
    rng = np.random.default_rng(4)
    n = 800
    A = dict(rv=np.exp(rng.normal(size=n) * 0.4), lnb=rng.normal(size=n) * 0.2, pi=rng.uniform(0.1, 0.9, n), bg=np.zeros(n), lnc8=rng.normal(size=n) * 0.1)
    T = np.arange(50, 700)
    c, m = ML.stats_l("pibar", A, T)
    for lam in (0.0, 1e-3, 1e-1):
        assert fit(A, lam, c, m, "pibar", T) == 0.0
    c1, m1 = ML.stats_l("pi", A, T)
    assert fit(A, 1e-3, c1, m1, "pi", T) != 0.0
