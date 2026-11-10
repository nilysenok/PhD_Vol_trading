"""Этап Z: синтетика с известным ответом (для тестов): кадр в формате x1.parquet (d, t_plus, rv, F = X1, s = e_d) и данные пар D (u, s, e, Z). rv = X1·exp(g_true)·ε, ε ~ Gamma(k)/k (E ε = 1) — минимум QLIKE в g_true,
поэтому (b, a) восстанавливаются. g_true = e_d·(b0 + a0·s_d) (поправка только на области). Сети и файлов нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE = dict(b0=0.2, a0=0.3)


def make(n: int = 600, seed: int = 0, truth: dict | None = TRUE, start: str = "2018-01-03", p_e: float = 0.15, shape: float = 3.0, noise: bool = True) -> tuple:
    """-> (x1 DataFrame, D dict: u, s, e, Z). truth=None — сигнала нет (rv = X1·ε)."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    s, e = rng.normal(0.0, 0.8, n), (rng.random(n) < p_e).astype(int)
    x, v = rng.normal(2.0, 0.4, n), rng.normal(0.0, 0.7, n)
    X1 = np.exp(rng.normal(-1.0, 0.3, n))
    g = np.zeros(n) if truth is None else e * (truth["b0"] + truth["a0"] * s)
    eps = rng.gamma(shape, 1.0 / shape, n) if noise else np.ones(n)
    rv = X1 * np.exp(g) * eps
    df = pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, M=X1, C=X1, s=e, pi=0.5, F=X1))
    D = dict(u=rv / X1, s=s, e=e, Z=np.column_stack([e.astype(float), x, s, v]))
    return df, D
