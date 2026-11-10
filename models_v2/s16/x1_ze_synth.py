"""Лист ZE: синтетика с известным ответом (для тестов). Кадр в формате x1.parquet (d, t_plus, rv, F = X1, s = e_d) и данные пар D (u, s, e, G, Z). rv = X1·exp(g_true)·ε, ε ~ Gamma(k)/k (E ε = 1) — минимум QLIKE в g_true,
поэтому (b, a, c) восстанавливаются. g_true = e_d·(b0 + a0·s_d + c0·G_d) (поправка только на области). G_d — внешний признак N(0,15; 0,3²). Сети и файлов нет."""
from __future__ import annotations

import numpy as np

from models_v2.s16 import x1_z_synth as SY

TRUE = dict(b0=0.1, a0=0.3, c0=0.5)


def make(n: int = 600, seed: int = 0, truth: dict | None = TRUE, p_e: float = 0.3, shape: float = 3.0, noise: bool = True) -> tuple:
    """-> (x1 DataFrame, D dict: u, s, e, G, Z). truth=None — сигнала нет (rv = X1·ε)."""
    x1, D = SY.make(n, seed, truth=None, p_e=p_e, shape=shape, noise=False)       # те же X1, s, e; rv пересчитаем с G
    rng = np.random.default_rng(seed + 1000)
    G = rng.normal(0.15, 0.3, n)
    X1, e, s = x1["F"].to_numpy(float), D["e"], D["s"]
    g = np.zeros(n) if truth is None else e * (truth["b0"] + truth["a0"] * s + truth["c0"] * G)
    rv = X1 * np.exp(g) * (rng.gamma(shape, 1.0 / shape, n) if noise else np.ones(n))
    return x1.assign(rv=rv), dict(D, u=rv / X1, G=G)
