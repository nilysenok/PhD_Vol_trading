"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 6.3): статистические процедуры оценки — чистые функции по массивам дневных потерь; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Потеря l = RV/F − ln(RV/F) − 1 (`v6_0_q.ql`); Δ = 100·mean(l_h − l_b)/mean(l_b), минус — гипотеза лучше.
Публичное API:
- константы `B = 10000`, `MEAN_BLOCK = 20`, `SEED = 0`, `ALPHA = 0.05`, `MCS_ALPHA = 0.10`.
- `nw_lag(n) = ceil(4·(n/100)^(2/9))` (NW-94; 8 при n = 2198); `LAG_REF = 5` (справочно, как W1–W8).
- `stat_idx(n, B_, mean_block, seed) -> (B_, n) int32`: стационарный бутстрап Politis–Romano (геометрическая длина блока, среднее `mean_block`, по кругу); `idx_for(n)` — то же с B/20/0, кэш.
- `dm_nw(l_base, l_model, lag) -> dict(mean_d, stat, p_one, lag, n)`: d = l_base − l_model, плюс = модель лучше; p_one — односторонний «модель лучше» (Ньюи–Уэст, Бартлетт).
- `sigma_lr(d, lag)`: √(n·nw_variance(d)) — долгосрочное σ ряда разностей потерь.
- `boot_ci(l_h, l_b, idx) -> dict(delta_pct, lo90, hi90, p_boot)`: 90 % перцентильный интервал и p_boot = (#{средних ≥ 0} + 1)/(B + 1).
- `holm_n(p, N)`: Холм с N испытаний; NaN-p ИСКЛЮЧАЮТСЯ до ранжирования и остаются NaN (не превращаются в 1 бегущим максимумом; `S_STAT_AUDIT.md` разд. 4); N < числа конечных p — ошибка.
- `finite_pos_mask(rv, F)`, `mcs(rv, F, idx=None) -> DataFrame(series, mean_qlike, p, in_mcs, n_used)`: MCS 90 % (T_max, `stats_d.mcs`) ТОЛЬКО на датах, где rv > 0 и все F конечны и > 0.
- `power(d, l_base_mean, lag) -> dict`: σ_LR, наблюдаемый эффект (абс., %), n для мощности 0,5 и 0,8 при α = 0,05 односторонний: ((z_α + z_β)·σ_LR/|Δ|)²; то же для гипотетического −1 % QLIKE базы; при эффекте не в пользу модели — NaN."""
from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q

B, MEAN_BLOCK, SEED, ALPHA, MCS_ALPHA, LAG_REF = 10000, 20, 0, 0.05, 0.10, 5


def nw_lag(n: int) -> int:
    return int(math.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def stat_idx(n: int, B_: int = B, mean_block: int = MEAN_BLOCK, seed: int = SEED) -> np.ndarray:
    """Стационарный бутстрап: на каждом шаге с вероятностью 1/mean_block — новый случайный старт, иначе следующая дата (по кругу)."""
    rng = np.random.default_rng(seed)
    idx = np.empty((B_, n), np.int32)
    idx[:, 0] = rng.integers(0, n, B_)
    for t in range(1, n):
        new = rng.random(B_) < 1.0 / mean_block
        idx[:, t] = np.where(new, rng.integers(0, n, B_), (idx[:, t - 1] + 1) % n)
    return idx


@lru_cache(maxsize=4)
def idx_for(n: int) -> np.ndarray:
    return stat_idx(n)


def dm_nw(l_base, l_model, lag: int) -> dict:
    d = np.asarray(l_base, float) - np.asarray(l_model, float)
    n = len(d)
    if n < 2:
        return dict(mean_d=float("nan"), stat=float("nan"), p_one=float("nan"), lag=lag, n=n)
    v, m = dmod.nw_variance(d, lag), float(d.mean())
    if not v > 0:
        return dict(mean_d=m, stat=float("nan"), p_one=float("nan"), lag=lag, n=n)
    z = m / math.sqrt(v)
    return dict(mean_d=m, stat=z, p_one=0.5 * math.erfc(z / math.sqrt(2)), lag=lag, n=n)


def sigma_lr(d, lag: int) -> float:
    d = np.asarray(d, float)
    return math.sqrt(len(d) * dmod.nw_variance(d, lag))


def boot_ci(l_h, l_b, idx: np.ndarray) -> dict:
    d = np.asarray(l_h, float) - np.asarray(l_b, float)
    bm = d[idx].mean(axis=1)
    lvl = float(np.mean(l_b))
    return dict(delta_pct=100 * float(d.mean()) / lvl, lo90=100 * float(np.percentile(bm, 5)) / lvl, hi90=100 * float(np.percentile(bm, 95)) / lvl,
                p_boot=float((np.sum(bm >= 0) + 1) / (len(bm) + 1)))


def holm_n(p, N: int) -> np.ndarray:
    """Холм (step-down, монотонный) с N испытаний программы; недостающие до N p = 1 неявно (множитель N − r). NaN исключены и остаются NaN."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    if N < int(ok.sum()):
        raise ValueError("N=%d меньше числа конечных p %d" % (N, int(ok.sum())))
    out, run = np.full(len(p), np.nan), 0.0
    idx = np.flatnonzero(ok)
    for r, i in enumerate(idx[np.argsort(p[idx], kind="stable")]):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def finite_pos_mask(rv, F: dict) -> np.ndarray:
    m = np.isfinite(rv) & (np.asarray(rv, float) > 0)
    for f in F.values():
        f = np.asarray(f, float)
        m &= np.isfinite(f) & (f > 0)
    return m


def mcs(rv, F: dict, idx: np.ndarray | None = None) -> pd.DataFrame:
    rv, names = np.asarray(rv, float), list(F)
    m = finite_pos_mask(rv, F)
    L = np.column_stack([Q.ql(rv[m], np.asarray(F[k], float)[m]) for k in names])
    r = SD.mcs(L, idx_for(len(L)) if idx is None else idx, alpha=MCS_ALPHA)
    return pd.DataFrame({"series": names, "mean_qlike": L.mean(axis=0), "p": r["p"], "in_mcs": r["in"], "n_used": int(m.sum())})


def _n_req(eff: float, sigma: float, zb: float) -> float:
    if not (eff < 0 and sigma > 0):
        return float("nan")
    return float(((NormalDist().inv_cdf(1 - ALPHA) + zb) * sigma / abs(eff)) ** 2)


def power(d, l_base_mean: float, lag: int) -> dict:
    """d = l_кандидат − l_база; эффект = mean(d) (минус — в пользу кандидата)."""
    d = np.asarray(d, float)
    eff, sig, z8 = float(d.mean()), sigma_lr(d, lag), NormalDist().inv_cdf(0.8)
    hyp = -0.01 * l_base_mean
    return dict(sigma_lr=sig, effect_abs=eff, effect_pct=100 * eff / l_base_mean, n_p05=_n_req(eff, sig, 0.0), n_p08=_n_req(eff, sig, z8),
                n_p05_hyp1pct=_n_req(hyp, sig, 0.0), n_p08_hyp1pct=_n_req(hyp, sig, z8))
