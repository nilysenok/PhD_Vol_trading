"""Проверки S-A-IMPL-2 как функции (честный код проходит, подложенные ошибки — нет). Не тест. Эталоны — независимый numpy по формулам разд. 3 спецификации."""
import numpy as np
import pandas as pd
import torch

from models_v2.s18 import a_feat as A
from models_v2.s18 import a_model as M
from s18_a_guard import ROOT

LAM = 1e-3
Z = torch.float64


def cfg8():
    return pd.concat([pd.read_parquet(ROOT / p) for p in ("data/fwd/models/cfg8.parquet", "data/fwd/models/new/cfg8.parquet")]).set_index("d")


def zero():
    return torch.zeros(M.NP, dtype=Z)


def check_nest(D):
    """θ = 0 ⇒ C_A = cfg8.F на 2198 датах, H_A = K1 на всех парах (1e-12), все варианты; пары без cfg8: C_A = H_A."""
    V = D.view(np.arange(len(D.d)))
    F8 = cfg8()["F"].reindex(D.d).to_numpy()
    assert D.has8.sum() == 2198 and np.isfinite(F8[D.has8]).all()
    for var in M.VARIANTS:
        lnH, lnC = M.forward(zero(), V, var)
        assert np.abs(np.exp(lnC.numpy()[D.has8]) / F8[D.has8] - 1).max() <= 1e-12, var
        assert np.abs(np.exp(lnH.numpy() - D.lnk1) - 1).max() <= 1e-12, var
        assert np.array_equal(lnC.numpy()[~D.has8], lnH.numpy()[~D.has8])


def ref_forward(D, x, var):
    """Независимый расчёт ln H_A, ln C_A по формулам 3.1–3.2 из полей Data (δ, ω, ψ, c, δw, δκ — вектор x)."""
    dl, w, kap = x[:5], x[26], x[27]
    if var == "A0":
        lnS = D.lnS0
    else:
        th = dict(omega=torch.as_tensor(x[5:8]), psi=torch.as_tensor(x[8:14].reshape(3, 2)), c=torch.as_tensor(x[14:26].reshape(3, 4)))
        with torch.no_grad():
            lnS = A.ln_S(D.F, th, A.lag_idx(D.F, D.rows), M.tau_of(var)).numpy()
    lnH = D.lnk1 + dl[0] + ((D.beta + dl[1:4]) * lnS).sum(1) - (D.beta * D.lnS0).sum(1) + dl[4] * D.ln1pj
    weff = np.clip(D.w + w, np.minimum(D.w, 0.0), np.maximum(D.w, 1.0))
    lnC = np.where(D.has8, lnH + weff * (D.lnb - lnH) + (1 + kap) * D.m, lnH)
    return lnH, lnC


def rand_x(seed):
    g = np.random.default_rng(seed)
    dwk = np.array([abs(g.normal(0, 0.05)) + 0.02, g.normal(0, 0.05)])      # δw > 0: иначе клип w_eff скрывает g у пар без cfg8
    return np.concatenate([g.normal(0, 0.05, 5), g.uniform(0.05, 0.9, 3), g.uniform(-2, 2, 6), g.normal(0, 0.4, 12), dwk])


def check_formula(D, fwd=None, seeds=(1, 2)):
    """forward = независимый эталон при ненулевых параметрах: ℓ по H_A (не по K1), m не G, пары без cfg8 — g = 0."""
    fwd = fwd or M.forward
    V = D.view(np.arange(len(D.d)))
    for var in M.VARIANTS:
        for s in seeds:
            x = rand_x(s)
            lnH, lnC = fwd(torch.as_tensor(x), V, var)
            rH, rC = ref_forward(D, x, var)
            assert np.abs(lnH.detach().numpy() - rH).max() <= 1e-10 and np.abs(lnC.detach().numpy() - rC).max() <= 1e-10, (var, s)


def check_penalty(pen=None):
    """Известный ответ: единичный вектор по блоку даёт λ (δ, ω, c, δw, δκ) или 1e-4 (ψ)."""
    pen = pen or M.penalty
    for sl, want in ((M.I_D, LAM), (M.I_W, LAM), (M.I_C, LAM), (M.I_DW, LAM), (M.I_DK, LAM), (M.I_P, 1e-4)):
        v = zero()
        v[sl] = 1.0
        n = 1.0 * len(v[sl].reshape(-1)) if isinstance(sl, slice) else 1.0
        assert abs(float(pen(v, LAM)) - want * n) <= 1e-15, sl


def check_grad(D, loss=None, n_points=2, seed=0):
    """autograd = центральные разности (h = 1e-6), отн. по норме ≤ 1e-6, A2, все 28 параметров, 450 пар (с cfg8 и без)."""
    loss = loss or M.loss
    V = D.view(np.arange(150, 600))
    for s in range(n_points):
        x = rand_x(seed + s)
        v = torch.as_tensor(x).clone().requires_grad_(True)
        loss(v, V, "A2", LAM).backward()
        ga = v.grad.numpy().copy()
        gn = np.empty(M.NP)
        for k in range(M.NP):
            e = np.zeros(M.NP)
            e[k] = 1e-6
            gn[k] = (float(loss(torch.as_tensor(x + e), V, "A2", LAM)) - float(loss(torch.as_tensor(x - e), V, "A2", LAM))) / 2e-6
        assert np.linalg.norm(ga - gn) / max(np.linalg.norm(gn), 1e-30) <= 1e-6, s


def check_starts():
    """seed 0 — ноль; seeds 1–4 — ненулевой старт на свободных ω, ψ, c (ω ∈ [0, 1]), попарно различны; несвободные = 0."""
    for var in ("A1", "A2", "A2p"):
        free = M.free_mask(var, True)
        x = [M._x0(var, s, free) for s in range(5)]
        assert not x[0].any() and not (x[1:][0] == x[1:][1]).all()
        for s in range(1, 5):
            assert np.abs(x[s][M.I_W]).max() > 0 and x[s][M.I_W].min() >= 0 and x[s][M.I_W].max() <= 1
            assert not x[s][~free].any()
        assert len({tuple(np.round(v, 12)) for v in x}) == 5
    assert not M._x0("A0", 3, M.free_mask("A0", True)).any()
