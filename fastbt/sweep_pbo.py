"""PBO по CSCV (Bailey, Borwein, López de Prado, Zhu) для движка перебора (Ж-034, Э1-а′ v1.1).

Перенесено из `sweep_stat.py`; там остался реэкспорт, чтобы `ss.pbo_cscv` продолжал работать.
"""
from __future__ import annotations

from itertools import combinations
from math import log

import numpy as np

PBO_BLOCKS = 8       # S = 8 блоков CSCV


def _sr_cols(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 1e-12, m.mean(axis=0) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def pbo_logits(m, s: int = PBO_BLOCKS) -> np.ndarray:
    """λ = ln(ω/(1−ω)) для всех C(s, s/2) разбиений IS/OOS; m — [T, N] доходности N точек.
    T режется на s равных блоков (хвост-остаток отбрасывается); ω — относительный OOS-ранг лучшей IS-точки (по Шарпу).
    N < 2 или T < 2s — пустой массив."""
    m = np.asarray(m, float)
    if s < 2 or s % 2:
        raise ValueError("число блоков CSCV — чётное ≥ 2")
    t, n = m.shape
    if n < 2 or t < 2 * s:
        return np.array([])
    L = t // s
    blocks = [np.arange(i * L, (i + 1) * L) for i in range(s)]
    lam = []
    for comb in combinations(range(s), s // 2):
        rest = [b for b in range(s) if b not in comb]
        is_i, oos_i = (np.concatenate([blocks[b] for b in g]) for g in (comb, rest))
        best = int(np.argmax(_sr_cols(m[is_i])))
        oos = _sr_cols(m[oos_i])
        w = ((oos < oos[best]).sum() + 1) / (n + 1)
        lam.append(log(w / (1 - w)))
    return np.array(lam)


def pbo_cscv(m, s: int = PBO_BLOCKS) -> float:
    """PBO = доля λ ≤ 0 (лучшая IS-точка в OOS не выше медианы). N < 2 или T < 2s — nan. Случайные данные — ≈ 0,5."""
    lam = pbo_logits(m, s)
    return float(np.mean(lam <= 0)) if len(lam) else float("nan")
