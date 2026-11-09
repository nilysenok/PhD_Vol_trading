"""V7-1a-M: синтетика для тестов и замера времени (НЕ данные SBER). Таблица `tab` той же формы, что у `v7_1m_data.load_tab`; цель RV = F_true·ε, ε ~ Gamma(k, 1/k) со средним 1 (QLIKE-оптимум = F_true).
Два режима (марковская цепь): знак поправки зависит от режима — G_true = ±A·tanh(z₁); режим виден в Z через ℓ = ln(M/H) (K2, K3 сдвинуты на режим), эксперты видят z."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1m_feat as FE

AMP, GAMMA_K = 0.6, 25.0


def make(n: int = 1100, seed: int = 0, start: str = "2016-12-30", amp: float = AMP, regimes: bool = True) -> dict:
    """-> {tab (с rv_d, jshare), Zg (n, 9) — настоящие признаки `build_Z`, z (n, 4), regime (n,), G_true (n,)}. n = 1100 строк ≈ 2017–2021 по порядку."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n + 1)
    reg = np.zeros(n, int)
    for i in range(1, n):
        reg[i] = reg[i - 1] if rng.random() > 0.04 else 1 - reg[i - 1]
    z = rng.standard_normal((n, 4))
    G_true = (np.where(reg == 1, 1.0, -1.0) if regimes else 1.0) * amp * np.tanh(z[:, 0])
    H = np.exp(rng.normal(0.0, 0.25, n).cumsum() * 0.05 + 1.0 + 0.3 * np.sin(np.arange(n) / 40.0))
    eps = rng.gamma(GAMMA_K, 1.0 / GAMMA_K, n)
    rv = H * np.exp(G_true) * eps
    K2 = H * np.exp(0.05 * rng.standard_normal(n) + 0.1 * (reg - 0.5))
    K3 = H * np.exp(0.05 * rng.standard_normal(n) + 0.1 * (reg - 0.5))
    tab = pd.DataFrame({"H": H, "K2": K2, "K3": K3, "rv_next": rv, "t_plus": d[1:], "rv_d": np.r_[rv[0], rv[:-1]], "jshare": rng.uniform(0, 0.3, n)}, index=d[:-1])
    tab.iloc[0, tab.columns.get_loc("rv_d")] = np.nan
    return dict(tab=tab, Zg=FE.build_Z(tab).to_numpy(float), z=z, regime=reg, G_true=G_true)


def perturb_future(s: dict, j: int, factor: float = 3.0) -> dict:
    """Копия синтетики, где цели строк ≥ j (rv_next) умножены на `factor`, rv_d и Z пересчитаны. Для PIT-тестов: всё, что зависит только от пар 0…j−1, обязано остаться прежним."""
    tab = s["tab"].copy()
    tab.iloc[j:, tab.columns.get_loc("rv_next")] *= factor
    rv = tab["rv_next"].to_numpy()
    tab["rv_d"] = np.r_[np.nan, rv[:-1]]
    return {**s, "tab": tab, "Zg": FE.build_Z(tab).to_numpy(float)}
