"""Спецификации строк B1: что обучается (X, y), как f переводится в σ̂² (kind), потеря, веса L5, сжатие c.

kind: 'ln'   — σ̂² = exp(off + c·f + s²/2)   (L2: ln-цель);
      'lnmu' — σ̂² = exp(off + c·f)          (gamma / qlike: f = ln μ на шкале y_ratio);
      'lvl'  — σ̂² = max(mult·f + add, ε)    (прогноз в уровне; ε — PIT-граница, отрицательное запрещено).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models_v2.s3 import resid
from models_v2.s5 import pit

Q5 = [.2, .4, .6, .8]
N_MIN = 300      # R2: минимум пар эксперта режима, иначе резерв — Б-2


@dataclass
class Spec:
    name: str
    loss: str
    X: np.ndarray
    y: np.ndarray
    kind: str
    off: np.ndarray | None = None
    mult: np.ndarray | None = None
    add: np.ndarray | None = None
    gw: float = 0.0
    cs: tuple = (1.0,)
    grp: np.ndarray | None = None     # R2: группа режима строки t (NaN — нет)
    ng: int = 0
    tag: dict = field(default_factory=dict)


def regime_cols(F) -> dict:
    """Режимные функции строки t по прошлому: q_w (квинтиль RV_w по 252 предыдущим), стресс_RV (RV_w > прошлый q67), переход q_t − q_{t−1}."""
    rvw = np.exp(F.X[:, 1])
    qw = pit.past_bucket(rvw, Q5)
    q67 = pit.q_prior_nan(rvw, 2 / 3)
    st = np.where(np.isnan(q67) | np.isnan(rvw), np.nan, (rvw > q67).astype(float))
    tr = np.concatenate([[np.nan], qw[1:] - qw[:-1]])
    return {"qw": qw, "stress": st, "trans": tr}


def weights_l5(F, gw: float, idx: np.ndarray) -> np.ndarray:
    """w_t ∝ RV_w,t^(−γ) (функция только строки t), нормировка на среднее по парам окна idx (не по всей выборке)."""
    w = np.exp(-gw * F.X[:, 1])
    w = np.where(np.isfinite(w), w, 1.0)
    return w / w[idx].mean()


def _tgt(F, base: np.ndarray) -> np.ndarray:
    return resid.target_minus(F, base)


def build_specs(F, DZ: dict, lb: np.ndarray) -> dict:
    m, r, lv = DZ["m_mkx"], DZ["resid_mkx"], DZ["level_mkx"]
    S, harb, z = np.exp(m.off), np.exp(lb), np.zeros(len(F.rv))
    y_m, y_h, rv1 = np.exp(m.y), np.exp(r.y), np.exp(_tgt(F, z))
    rg = regime_cols(F)
    R1 = [np.column_stack([m.X, rg["qw"]]), np.column_stack([m.X, rg["qw"], rg["stress"], rg["trans"]]),
          np.column_stack([r.X, rg["qw"], rg["stress"], rg["trans"]])]
    sp = [
        Spec("ctrl_b2", "l2", m.X, m.y, "ln", m.off),
        Spec("ctrl_b1", "l2", r.X, r.y, "ln", lb, cs=(1.0, 0.75, 0.5)),
        Spec("obj_l1n", "mse", m.X, y_m, "lvl", mult=S, add=0 * S),
        Spec("obj_l1raw", "mse", lv.X, rv1, "lvl", mult=1 + z, add=z),
        Spec("obj_l3x", "gamma", m.X, y_m, "lnmu", m.off),
        Spec("obj_l3l", "lgb_gamma", m.X, y_m, "lnmu", m.off),
        Spec("obj_l4", "qlike", m.X, y_m, "lnmu", m.off),
        Spec("obj_l5a", "l2", m.X, m.y, "ln", m.off, gw=0.5),
        Spec("obj_l5b", "l2", m.X, m.y, "ln", m.off, gw=1.0),
        Spec("obj_l5c", "qlike", m.X, y_m, "lnmu", m.off, gw=0.5),
        Spec("har_h2_l4", "qlike", r.X, y_h, "lnmu", lb, cs=(1.0, 0.5)),
        Spec("har_h2_l4m", "qlike", m.X, y_h, "lnmu", lb),
        Spec("har_h1_s", "mse", r.X, y_h - 1.0, "lvl", mult=harb, add=harb),
        Spec("har_h1_raw", "mse", r.X, rv1 - harb, "lvl", mult=1 + z, add=harb),
        Spec("reg_r1a", "l2", R1[0], m.y, "ln", m.off),
        Spec("reg_r1b", "l2", R1[1], m.y, "ln", m.off),
        Spec("reg_r1c", "l2", R1[2], r.y, "ln", lb),
        Spec("reg_r2a", "l2", m.X, m.y, "ln", m.off, grp=_grp(rg["qw"], (3,)), ng=2),
        Spec("reg_r2b", "l2", m.X, m.y, "ln", m.off, grp=_grp(rg["qw"], (2, 4)), ng=3)]
    return {s.name: s for s in sp}


def _grp(qw: np.ndarray, cuts: tuple) -> np.ndarray:
    """Группа = число порогов cuts, которые q_w достиг (R2a: q<3 / q>=3; R2b: q<2 / 2..3 / q=4)."""
    g = sum((qw >= c).astype(float) for c in cuts)
    return np.where(np.isnan(qw), np.nan, g)


def sigma2(spec: Spec, rows, f, s2, eps, c: float = 1.0) -> tuple:
    """(σ̂², число сырых ≤ 0, число срабатываний границы) для строк rows."""
    rows = np.asarray(rows)
    f = np.asarray(f, float)
    if spec.kind == "ln":
        return np.exp(spec.off[rows] + c * f + np.asarray(s2, float) / 2), 0, 0
    if spec.kind == "lnmu":
        return np.exp(spec.off[rows] + c * f), 0, 0
    raw = spec.mult[rows] * f + spec.add[rows]
    out = np.maximum(raw, eps[rows])
    return out, int((raw <= 0).sum()), int((raw < eps[rows]).sum())
