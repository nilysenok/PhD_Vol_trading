"""V7 лист 1a-M, G16: формы прогноза. g ≡ 0 ⇒ F = H побитно (MoE A/B/C, якорь при w = 0, плоский); F > 0 на крайних входах; состав входов маршрутизатора (мутации «z в A», «Z в C»);
числа параметров A 51, B 67, C 31, эксперты 10, плоский 76. Синтетика, данных SBER нет."""
from __future__ import annotations

import pytest
import torch

from models_v2.s14 import v7_1m_forms as FM

DEC = dict(A=51, B=67, C=31)


def rn(*shape, seed=0):
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def inputs(n=48):
    return rn(n, 9, seed=1), rn(n, 4, seed=2), 0.3 * rn(n, seed=3), torch.exp(rn(n, seed=4))


def models():
    return [("A", FM.ResMoE("A")), ("B", FM.ResMoE("B")), ("C", FM.ResMoE("C")), ("anchor_w0", FM.ResMoE("B", anchor=True)), ("flat", FM.Flat()), ("flat_z", FM.Flat(FM.DZ)), ("K3", FM.ResMoE("B", K=3))]


@pytest.mark.parametrize("name,m", models())
def test_g_zero_gives_F_equals_H_bitwise(name, m):
    Zg, z, ell, H = inputs()
    G = m(Zg, z, ell)[0]
    assert torch.equal(G, torch.zeros_like(G))
    assert torch.equal(FM.forecast(H, G), H)


def test_anchor_with_w_nonzero_moves_and_g_zero_experts_only_anchor():
    Zg, z, ell, H = inputs()
    m = FM.ResMoE("B", anchor=True)
    with torch.no_grad():
        m.w.fill_(0.5)
    F = FM.forecast(H, m(Zg, z, ell)[0])
    assert torch.allclose(F, H * torch.exp(0.5 * ell))                  # эксперты нулевые: G = w·ℓ


@pytest.mark.parametrize("name,m", models())
def test_F_positive_at_extreme_inputs(name, m):
    Zg, z, ell, H = inputs()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(5 * rn(*p.shape, seed=7))
    for scale in (1e-300, 1.0, 1e6, 1e30):
        F = FM.forecast(H, m(scale * Zg, scale * z, ell)[0])
        assert bool(torch.isfinite(F).all()) and bool((F > 0).all()), (name, scale)


def test_param_counts_declared():
    c = FM.counts()
    assert c["router"] == DEC and c["experts"] == 10 and c["flat"] == 76 and c["moe_b"] == 77 and c["anchor_b"] == 78
    # K = 3 по формуле плана (проекция 13×4 = 52, выход 4×(4+1) = 20, эксперты 3×5 = 15) — 87; в плане п. 4 стоит 93 (расхождение в тексте плана, см. отчёт листа)
    assert FM.n_params(FM.ResMoE("B", K=3)) == 52 + 4 * 5 + 3 * 5 == 87


def test_router_param_split_matches_plan_formula():
    for k, din in (("A", 9), ("B", 13), ("C", 4)):
        m = FM.ResMoE(k)
        assert m.proj.weight.shape == (4, din) and m.proj.bias is None and m.out.weight.shape == (3, 4)
        assert FM.n_params(m.proj) + FM.n_params(m.out) == DEC[k]


def test_expert_is_bounded_tanh_form():
    m = FM.ResMoE("B")
    with torch.no_grad():
        m.a.copy_(torch.tensor([100.0, -100.0], dtype=torch.float64))
    g = m.experts(rn(10, 4))
    assert bool((g.abs() <= FM.C + 1e-12).all()) and torch.allclose(g[:, 0], torch.ones(10, dtype=torch.float64))


def _depends(m, kind_cols):
    """Какие группы входа меняют π: возмущаем Z / z по отдельности."""
    Zg, z, ell, H = inputs()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(rn(*p.shape, seed=9))
    base = m.pi(Zg, z)
    return dict(Z=not torch.equal(base, m.pi(Zg + 1.0, z)), z=not torch.equal(base, m.pi(Zg, z + 1.0)))


@pytest.mark.parametrize("kind,expect", [("A", dict(Z=True, z=False)), ("B", dict(Z=True, z=True)), ("C", dict(Z=False, z=True))])
def test_router_gets_exactly_declared_inputs(kind, expect):
    assert _depends(FM.ResMoE(kind), kind) == expect


@pytest.mark.parametrize("kind,wrong", [("A", lambda Zg, z: torch.cat([Zg, z], 1)), ("C", lambda Zg, z: torch.cat([Zg, z], 1)[:, :4]), ("B", lambda Zg, z: Zg)])
def test_mutation_router_input_composition_caught(monkeypatch, kind, wrong):
    """Мутанты «z в A» (13 столбцов в A), «Z в C» (первые столбцы Z вместо z), «B без z» ловятся составом/формой."""
    monkeypatch.setattr(FM, "router_input", lambda k, Zg, z: wrong(Zg, z))
    m = FM.ResMoE(kind)
    Zg, z, ell, H = inputs()
    try:
        ok = _depends(m, kind) == {"A": dict(Z=True, z=False), "B": dict(Z=True, z=True), "C": dict(Z=False, z=True)}[kind]
    except RuntimeError:                                                  # не сошлись размеры матрицы — тоже поймано
        ok = False
    assert not ok


def test_unknown_router_refused():
    with pytest.raises(ValueError):
        FM.router_input("D", *inputs()[:2])


def test_deterministic_context_restores_process_state():
    before = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    with FM.deterministic():
        assert torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 1
    assert (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()) == before


def test_same_seed_same_init_different_seed_differs():
    a, b, c = FM.ResMoE("B", seed=3), FM.ResMoE("B", seed=3), FM.ResMoE("B", seed=4)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a.proj.weight, c.proj.weight)
