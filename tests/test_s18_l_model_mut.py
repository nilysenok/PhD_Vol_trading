"""S-L-IMPL-1 (`S_L_SPEC.md` разд. 6, мутации модели): каждая подложенная ошибка роняет детектор, честный код его проходит (№ 164, № 103-класс «штраф не передан», граница, m := 1, сдвиг без переноса)."""
import numpy as np

from models_v2.s18 import l_model as ML
from models_v2.s18 import r_loss as RL


def data(n=800, seed=4):
    rng = np.random.default_rng(seed)
    return dict(rv=np.exp(rng.normal(size=n) * 0.4), lnb=rng.normal(size=n) * 0.2, pi=rng.uniform(0.1, 0.9, n), bg=np.zeros(n), lnc8=rng.normal(size=n) * 0.1)


def opt(a_true, lam=0.0):
    n = 400
    z = np.where(np.arange(n) % 2 == 0, 1.4, 0.6)
    A = dict(rv=np.exp(a_true) * z, lnb=np.zeros(n), pi=np.ones(n), bg=np.zeros(n), lnc8=np.zeros(n))
    return ML.fit_kind("pi", A, np.arange(n), lam, 1e6, 1.0)


# детекторы: пустой список = честно
def det_stats_m(stats) -> list:
    A, T = data(), np.arange(50, 700)
    pt = ML.train_pi("pi", A["pi"], T)
    c, m = RL.clip_stats(A["rv"][T] / np.exp((1 - pt) * A["lnb"][T] + pt * A["lnc8"][T] + A["bg"][T]), np.arange(len(T)))
    return [] if np.allclose(stats("pi", A, T), (c, m), rtol=1e-12, atol=0) else ["c, m ≠ clip_stats"]


def det_shift(train_pi) -> list:
    pi, idx = np.random.default_rng(2).uniform(size=2198), np.arange(2198)
    return [] if np.array_equal(train_pi("shift252", pi, idx), np.roll(pi, 252)[idx]) and np.array_equal(train_pi("shift252", pi, idx), pi[(idx - 252) % 2198]) else ["сдвиг ≠ np.roll(pi, 252)"]


def det_bound() -> list:
    return [] if abs(opt(0.8)) <= 0.5 + 1e-12 and abs(opt(-0.8)) <= 0.5 + 1e-12 else ["â вне [−0,5; 0,5]"]


def det_penalty() -> list:
    return [] if abs(opt(0.3, 1e-1)) < abs(opt(0.3, 0.0)) - 1e-3 else ["λ·a² не действует"]


def det_ctrl_a(stats) -> list:
    A, T = data(), np.arange(50, 700)
    c, m = stats("pibar", A, T)
    return [] if ML.fit_kind("pibar", A, T, 1e-3, c, m) == 0.0 else ["rl1a: â ≠ 0"]


def test_honest_passes_every_detector():
    assert det_stats_m(ML.stats_l) == det_shift(ML.train_pi) == det_bound() == det_penalty() == det_ctrl_a(ML.stats_l) == []


def test_mutation_m_equals_one_is_caught():
    assert det_stats_m(lambda k, A, T: (ML.stats_l(k, A, T)[0], 1.0))


def test_mutation_shift_one_row_or_without_wrap_is_caught():
    one = lambda kind, pi, T: np.roll(np.asarray(pi, float), 1)[np.asarray(T)]
    nowrap = lambda kind, pi, T: np.concatenate([np.full(252, np.asarray(pi)[0]), np.asarray(pi)[:-252]])[np.asarray(T)]
    assert det_shift(one) and det_shift(nowrap)


def test_mutation_bound_removed_is_caught(monkeypatch):
    monkeypatch.setattr(ML, "A_BOUND", 5.0)
    assert det_bound() == ["â вне [−0,5; 0,5]"]


def test_mutation_penalty_not_passed_is_caught(monkeypatch):
    orig = ML.loss_l
    monkeypatch.setattr(ML, "loss_l", lambda a, rv, lnb, pi_t, bg, lnc8, lam, c, m: orig(a, rv, lnb, pi_t, bg, lnc8, 0.0, c, m))
    assert det_penalty()


def test_mutation_control_a_c_m_from_rv_over_w5g_is_caught():
    assert det_ctrl_a(lambda k, A, T: ML.stats_l("pi", A, T))
