"""D2: диагностики финалистов — leave-one-year-out, топ-20 вкладов, нижний дециль RV по прошлому, накопленная разность потерь."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import pit


def loyo_years(rv, s2_cfg, s2_h, s2_b, td) -> pd.DataFrame:
    """Для каждого года: описание без этого года — QLIKE, Δ к HAR-J(б) и Б-2 (минус = конфиг лучше)."""
    y = pd.Series(td).dt.year.to_numpy()
    qc, qh, qb = (metrics.qlike(rv, s) for s in (s2_cfg, s2_h, s2_b))
    rec = []
    for k in sorted(set(y)):
        m = y != k
        rec.append({"year_excluded": int(k), "n": int(m.sum()), "qlike": float(qc[m].mean()), "d_harj": float((qc - qh)[m].mean()), "d_b2": float((qc - qb)[m].mean())})
    return pd.DataFrame(rec)


def top20(rv, s2_cfg, s2_base, td, k: int = 20) -> pd.DataFrame:
    """k дней с наибольшей |Δ потерь| (конфиг − база): дата, Δ, доля в Σ Δ; итоговые доли в атрибутах."""
    dl = metrics.qlike(rv, s2_cfg) - metrics.qlike(rv, s2_base)
    o = np.argsort(-np.abs(dl))[:k]
    tot = float(dl.sum())
    df = pd.DataFrame({"target_date": pd.Series(td).iloc[o].dt.strftime("%Y-%m-%d").to_numpy(), "delta": dl[o], "share_of_sum": dl[o] / tot if tot else np.nan})
    df.attrs.update(sum_all=tot, sum_top=float(dl[o].sum()), sum_abs_share=float(np.abs(dl[o]).sum() / np.abs(dl).sum()), mean_without_top=float(np.delete(dl, o).mean()))
    return df


def low_rv_thr(x: np.ndarray, q: float = 0.1, include_current: bool = False) -> np.ndarray:
    """Порог на день k: квантиль q по 252 предыдущим RV_w (без дня k). include_current=True — мутация MD13 (№ 22): порог зависит от x[k]."""
    x = np.asarray(x, float)
    if not include_current:
        return pit.q_prior_nan(x, q)
    thr = np.full(len(x), np.nan)
    for k in range(pit.WIN, len(x)):
        thr[k] = np.nanquantile(x[k - pit.WIN + 1:k + 1], q)
    return thr


def low_rv_mask(rv_w_full: np.ndarray, grid, q: float = 0.1) -> np.ndarray:
    """Дни сетки, где RV_w(d) <= квантиль q по 252 предыдущим (без текущего дня)."""
    x = np.asarray(rv_w_full, float)
    thr = low_rv_thr(x, q)
    g = np.asarray(grid, int)
    return np.isfinite(thr[g]) & np.isfinite(x[g]) & (x[g] <= thr[g])


def low_rv_table(rv, s2: dict, mask: np.ndarray) -> pd.DataFrame:
    """Качество в нижнем дециле: n, QLIKE, RV/σ̂² (среднее), Δ к 'harj' и 'b2' (минус = лучше)."""
    q = {k: metrics.qlike(rv, v) for k, v in s2.items()}
    return pd.DataFrame([{"model": k, "n": int(mask.sum()), "qlike": float(q[k][mask].mean()), "mean_r": float((rv / s2[k])[mask].mean()),
                          "d_harj": float((q[k] - q["harj"])[mask].mean()), "d_b2": float((q[k] - q["b2"])[mask].mean())} for k in s2])
