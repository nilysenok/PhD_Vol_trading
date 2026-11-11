"""Синтетика для тестов S-C-IMPL-2 (не тест): пары (d, t⁺) с известной структурой 𝓜_K. Только numpy/torch, сети нет."""
import numpy as np
import torch

from models_v2.s18 import cm_model as M

THETA_STAR = np.array([0.4, 0.6, 0.1, -0.1, -0.3, 0.2])        # (γ_c, γ_p, θ_c0, θ_p0, θ_c1, θ_p1), разд. 7 M.G3


def make_data(n=300, seed=0, theta=None, shape=60.0, sc_range=(0.2, 0.5), su_range=(0.1, 0.3), random_share=False, cfg8=None):
    """Data на n парах: ŝ_c + ŝ_p + ŝ_u = 1 (ŝ_u > 0), Ĥ_X = A_X·exp(0,5·ξ); rv = cfg8·𝓜(θ)·Gamma(shape)/shape (среднее 1)."""
    g = np.random.default_rng(seed)
    sc = g.uniform(*sc_range, n)
    su = g.uniform(*su_range, n)
    sp = 1.0 - sc - su
    c8 = np.exp(-8.5 + 0.3 * g.normal(size=n)) if cfg8 is None else np.asarray(cfg8, float)
    Hc, Hp = sc * c8 * np.exp(0.5 * g.normal(size=n)), sp * c8 * np.exp(0.5 * g.normal(size=n))
    zc, zp = 0.5 * g.normal(size=n), 0.5 * g.normal(size=n)
    base = dict(rv=c8, c8=c8, Hc=Hc, Hp=Hp, sc=sc, sp=sp, su=su, zc=zc, zp=zp, labc=sc * c8, labp=sp * c8)
    D = M.Data(**base)
    if theta is not None:
        V = D.view(np.arange(n))
        Mv = M.forward(torch.as_tensor(np.asarray(theta, float)), V, "CK")["M"].numpy()
        eps = g.gamma(shape, 1.0 / shape, n)
        labc = g.gamma(shape, 1.0 / shape, n) * sc * c8 * np.exp(0.5 * g.normal(size=n)) if random_share else sc * c8
        base.update(rv=c8 * Mv * eps, labc=labc)
        D = M.Data(**base)
    return D


def params(seed, scale=1.0):
    return torch.as_tensor(np.random.default_rng(seed).uniform(-1, 1, M.NP) * scale)
