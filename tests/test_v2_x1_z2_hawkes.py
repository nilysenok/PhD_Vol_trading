"""Z2: ступень 1 — численный эталон по формулам плана (медленный), аналитический градиент = численному, синтетический Hawkes восстанавливается, Пуассон -> n̂ ≈ 0, мутации самовозбуждения и порядка рекурсии."""
import inspect
import math

import numpy as np
import pytest
from numba import njit
from scipy.optimize import approx_fprime

from models_v2.s16 import x1_z2_hawkes as HK, x1_z2_synth as SY
from models_v2.s16.x1_z2_ev import DELTA, NS


def ref_ll(lnmu, alpha, beta, eta, rho, N, m, pres, f, inlik, dg) -> float:
    """Эталон: рекурсия плана 2.2 прямыми циклами (бары < k возбуждают бар k; слот без бара — только затухание)."""
    mu, E, H, ll = math.exp(lnmu), math.exp(-beta * DELTA), 0.0, 0.0
    for j in range(len(N)):
        if j:
            H *= rho ** (1.0 + dg[j])
        for k in range(NS):
            ev = bool(pres[j, k] and N[j, k] > 0.5)
            if pres[j, k] and inlik[j]:
                x = DELTA * (mu * f[k] + H)
                ll += math.log(1.0 - math.exp(-x)) if ev else -x
            H = E * H + (alpha * math.exp(eta * m[j, k]) if ev else 0.0)
    return ll


def rough_ctx(seed=0, n=40):
    sim = SY.simulate(0.003, 0.5, math.log(2) / 60, 0.5, 0.5, n, seed=seed, f=np.linspace(0.5, 1.5, NS))
    ctx = sim["ctx"]
    ctx["pres"] = ctx["pres"].copy()
    ctx["pres"][3:6, 10:14] = False                      # пропущенные бары
    ctx["N"] = ctx["N"] * ctx["pres"]
    return ctx, sim["f"]


def test_kernel_equals_plan_reference_with_absent_slots_gaps_and_warmup():
    ctx, f = rough_ctx()
    W = HK.rows(ctx, 30, 40, 10)
    par = (math.log(0.003), 0.004, math.log(2) / 70, 0.7, 0.4)
    g = np.zeros(5)
    ll = HK.kernel(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"], DELTA, g)
    assert ll == pytest.approx(ref_ll(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"]), rel=1e-10)
    assert W["N"].sum() > 5 and not W["inlik"][:10].any()


def test_analytic_gradient_equals_numeric_including_n_beta_eta_rho_chain():
    ctx, f = rough_ctx(1)
    W = HK.rows(ctx, 30, 40, 10)
    for x in (HK.start(W), np.array([-6.0, 0.6, 0.004, 1.2, 0.8]), np.array([-5.0, 0.2, 0.02, 0.1, 0.0])):
        a = HK.nll(x, W, f)[1]
        num = approx_fprime(x, lambda y: HK.nll(y, W, f)[0], 1e-7)
        assert np.allclose(a, num, rtol=2e-4, atol=1e-8), (a, num)


def grad_alive(W, f) -> bool:
    """В старте плана градиент по КАЖДОМУ из пяти параметров ≠ 0 (мёртвая точка α = 0 исключена)."""
    return bool(np.all(np.abs(HK.nll(HK.start(W), W, f)[1]) > 1e-9))


def test_start_gradient_nonzero_and_dead_point_mutation_caught(monkeypatch):
    ctx, f = rough_ctx(2)
    W = HK.rows(ctx, 30, 40, 10)
    assert grad_alive(W, f)
    x = HK.start(W)
    x[1] = 0.0
    g = HK.nll(x, W, f)[1]                               # n = 0: β, η, ρ не влияют — градиент 0
    assert np.allclose(g[2:], 0.0, atol=1e-12)
    monkeypatch.setattr(HK, "N0", 0.0)
    assert not grad_alive(W, f)                          # мутант «старт n₀ = 0» ловится


def _mutant(old: str, new: str):
    src = inspect.getsource(HK._kernel)
    assert old in src
    ns = {"math": math}
    exec(src.replace(old, new), ns)
    return njit(ns["_kernel"])


def test_mutation_self_excitation_bar_k_excites_bar_k_caught():
    ctx, f = rough_ctx(3)
    W = HK.rows(ctx, 30, 40, 10)
    par = (math.log(0.003), 0.004, math.log(2) / 70, 0.7, 0.4)
    run = lambda k: k(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"], DELTA, np.zeros(5))
    mut = _mutant("lam = mu * f[k] + H ", "lam = mu * f[k] + H + alpha * ev * math.exp(eta * m[j, k]) ")
    ref = ref_ll(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"])
    assert run(HK.kernel) == pytest.approx(ref, rel=1e-10)
    assert abs(run(mut) - ref) > 1e-6


def test_mutation_no_overnight_decay_caught():
    ctx, f = rough_ctx(4)
    W = HK.rows(ctx, 30, 40, 10)
    par = (math.log(0.003), 0.004, math.log(2) / 70, 0.7, 0.4)
    mut = _mutant("pw, dpw = rho ** (1.0 + dg[j]), (1.0 + dg[j]) * rho ** dg[j]", "pw, dpw = 1.0, 0.0")
    ref = ref_ll(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"])
    assert abs(mut(*par, W["N"], W["m"], W["pres"], f, W["inlik"], W["dg"], DELTA, np.zeros(5)) - ref) > 1e-6


def test_known_answer_hawkes_recovered_and_poisson_gives_n_zero():
    true = dict(mu=0.002, n=0.5, beta=math.log(2) / 90, eta=0.5, rho=0.6)
    sim = SY.simulate(**true, n_days=2500, seed=11)
    r = HK.fit(SY.window_of(sim["ctx"]), sim["f"])
    assert abs(r["n"] - true["n"]) < 0.12 and abs(r["mu"] / true["mu"] - 1) < 0.3
    assert 0.6 < r["beta"] / true["beta"] < 1.7 and abs(r["rho"] - true["rho"]) < 0.3
    lr = HK.lr_test(r["ll"], r["ll0"])
    assert lr["p_strict"] < 1e-6 and lr["lr"] > 50
    po = SY.simulate(0.002, 0.0, math.log(2) / 90, 0.5, 0.6, 2500, seed=12)
    r0 = HK.fit(SY.window_of(po["ctx"]), po["f"])
    assert r0["n"] < 0.12 and HK.lr_test(r0["ll"], r0["ll0"])["p_strict"] > 0.01
