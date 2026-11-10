"""Y-A: синтетика с известным ответом (для тестов): кадр в формате x1.parquet (d, t_plus, rv, F = X1, s = e_d) и признаки дня. rv = X1·exp(g_true)·ε, ε ~ Gamma(k)/k (E ε = 1) — минимум QLIKE достигается в g_true,
поэтому параметры (a_n, a_e, κ) восстанавливаются. Сети и файлов нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE = dict(a_n=0.4, a_e=0.3, kappa=0.6)


def make(n: int = 600, seed: int = 0, truth: dict | None = TRUE, start: str = "2018-01-03", shape: float = 3.0) -> tuple:
    """-> (x1 DataFrame, ft dict: s, delta, zero, n, nlast, rv_d). truth=None — сигнала нет (rv = X1·ε)."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    tp = pd.DatetimeIndex(list(d[1:]) + [d[-1] + pd.offsets.BDay(1)])
    s = rng.normal(0.0, 0.8, n)
    e = (rng.random(n) < 0.15).astype(int)
    delta = ((tp - d).days.to_numpy() - 1).astype(float)
    x1 = np.exp(rng.normal(-1.0, 0.3, n))
    g = 0.0 if truth is None else (truth["a_n"] * (1 - e) + truth["a_e"] * e) * s * np.exp(-truth["kappa"] * delta)
    rv = x1 * np.exp(g) * rng.gamma(shape, 1.0 / shape, n)
    df = pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, M=x1, C=x1, s=e, pi=0.5, F=x1, pi0=0.5, pi1=0.5, a0=0.02, a1=0.02))
    ft = dict(s=s, delta=delta, zero=np.zeros(n, bool), n=np.full(n, 53), nlast=np.full(n, 6), rv_d=np.ones(n))
    return df, ft
