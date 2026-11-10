"""Y-A (Ж-243), обучение: аналитический градиент = численному; κ оживает после a; синтетика с известным ответом (a_n, a_e, κ восстанавливаются); выбор λ (ничья — больший; валидация только по закрытым парам,
мутация «λ по парам после D» ловится); Y2 с κ ≠ 0 ловится."""
from __future__ import annotations

import numpy as np
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_fit as FT, x1_ya_synth as SY


def arrays(n: int = 4000, seed: int = 0, truth=SY.TRUE):
    x1, ft = SY.make(n, seed, truth)
    return x1["rv"].to_numpy() / x1["F"].to_numpy(), x1["s"].to_numpy(float), np.asarray(ft["s"], float), np.asarray(ft["delta"], float)


@pytest.mark.parametrize("mode", ["Y1", "Y2"])
def test_analytic_gradient_equals_numeric(mode) -> None:
    u, e, s, dl = arrays(500)
    rng = np.random.default_rng(1)
    for _ in range(4):
        th = np.concatenate([rng.normal(0, 0.5, 2), rng.uniform(0, 3, 1)])[: FT.MODES[mode]]
        L, g = FT.loss_grad(th, u, e, s, dl, 0.01, mode)
        num = np.array([(FT.loss_grad(th + h, u, e, s, dl, 0.01, mode)[0] - FT.loss_grad(th - h, u, e, s, dl, 0.01, mode)[0]) / 2e-6 for h in np.eye(len(th)) * 1e-6])
        assert np.max(np.abs(g - num)) < 1e-6


def test_start_gradient_nonzero_for_a_and_kappa_alive_only_after_a() -> None:
    u, e, s, dl = arrays(2000)
    _, g0 = FT.loss_grad([0.0, 0.0, FT.KAPPA0], u, e, s, dl, 0.0, "Y1")
    assert abs(g0[0]) > 1e-4 and abs(g0[1]) > 1e-5          # градиент a в старте не нулевой
    assert g0[2] == 0.0                                      # при a = 0 g ≡ 0 и κ не влияет: мёртвая точка по κ
    _, g1 = FT.loss_grad([0.05, 0.05, FT.KAPPA0], u, e, s, dl, 0.0, "Y1")
    assert abs(g1[2]) > 1e-6                                 # после a κ оживает


def test_known_answer_parameters_recovered() -> None:
    u, e, s, dl = arrays(40000)
    th = FT.fit(u, e, s, dl, 0.0, "Y1")
    assert th[0] == pytest.approx(SY.TRUE["a_n"], abs=0.05) and th[1] == pytest.approx(SY.TRUE["a_e"], abs=0.05) and th[2] == pytest.approx(SY.TRUE["kappa"], abs=0.1)
    th2 = FT.fit(u, e, s, dl, 0.0, "Y2")
    assert len(th2) == 2 and th2[0] < th[0]                  # без затухания амплитуда в среднем меньше (Δ > 0 гасят перенос)
    assert FT.qlike_mean(th, u, e, s, dl, "Y1") < FT.qlike_mean(th2, u, e, s, dl, "Y2") < FT.qlike_mean([0, 0], u, e, s, dl, "Y2")


def test_no_signal_gives_zero_amplitudes_and_bounds_hold() -> None:
    u, e, s, dl = arrays(3000, truth=None)
    th = FT.fit(u, e, s, dl, 0.1, "Y1")
    assert np.max(np.abs(th[:2])) < 0.05 and 0.0 <= th[2] <= FT.BOUND_K
    z = FT.fit(u, e, np.zeros_like(s), dl, 0.0, "Y1")        # s ≡ 0: g ≡ 0, a остаётся в старте
    assert z[0] == 0.0 and z[1] == 0.0


def test_n_valid_boundary() -> None:
    assert [FT.n_valid(n) for n in (124, 251, 252, 253, 2000)] == [62, 125, 126, 126, 126]


def test_tie_takes_larger_lambda() -> None:
    u, e, s, dl = arrays(400)
    pool = np.arange(300)
    assert FT.choose_lambda(pool, u, e, np.zeros_like(s), dl, "Y1") == max(FT.LAMS)       # s ≡ 0: все λ дают одну потерю -> больший λ


def lam_case(honest: bool = True):
    """Закрытые пары: u = exp(0,3·s) (сигнал; лучший λ = 0); пары ПОСЛЕ них: u = exp(−0,3·s) (противоположный знак)."""
    rng = np.random.default_rng(0)
    n, c = 700, 400
    s = rng.normal(0, 1, n)
    e = np.zeros(n)
    g = np.where(np.arange(n) < c, 0.3, -0.3) * s
    return np.exp(g), e, s, np.zeros(n), np.arange(c)


def test_lambda_uses_only_closed_pairs_and_future_mutation_is_caught(monkeypatch) -> None:
    u, e, s, dl, pool = lam_case()
    lam = FT.choose_lambda(pool, u, e, s, dl, "Y2")
    u2 = u.copy()
    u2[len(pool):] = np.exp(rng_future(len(u) - len(pool)))   # любые будущие значения
    assert FT.choose_lambda(pool, u2, e, s, dl, "Y2") == lam == 0.0
    monkeypatch.setattr(FT, "valid_block", lambda p, v: np.arange(p[-1] + 1, p[-1] + 1 + v))          # мутация: валидация по парам после D
    assert FT.choose_lambda(pool, u, e, s, dl, "Y2") != lam


def rng_future(k: int):
    return np.random.default_rng(5).normal(0, 2, k)
