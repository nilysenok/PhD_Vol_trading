"""Ж-214, лист 3, V11: согласование прогнозов. Простая сумма положительных компонент (σ̂²_Σ из V7) — 0 trials. R2x `geo_blend`: прямой A0 (f_д) и σ̂²_Σ (f_Σ) согласуются геометрически
f = f_д^(1−w)·f_Σ^w, w ∈ {0; 0,1; …; 1} по ЗАКРЫТЫМ прогнозам (target_date <= d; >= 252 закрытых, иначе w = 0 — база A0); ничья (|Δ| <= 1e-12) — w ближе к 0,5, затем меньший (У17).
Компоненты масштабируются пропорционально к согласованной сумме (top-down): положительность и сумма сохраняются; при Σ <= 0 или нефинитной — деление поровну (флаг)."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s8 import hs_pit
from models_v2.s9 import hv2_dec as DC, hv2_world as W

GRID = np.round(np.linspace(0.0, 1.0, 11), 1)
MIN_CLOSED = 252


def blend(f_d, f_s, rv_t, d, td, extra: int = 0) -> tuple:
    """-> (согласованная сумма, w по дням). `extra` > 0 — МУТАНТ «закрыто + extra» (в выбор w попадают ещё не закрытые цели)."""
    f_d, f_s = np.asarray(f_d, float), np.asarray(f_s, float)
    m = hs_pit.closed_count(d, td, extra)
    cum = np.array([np.concatenate([[0.0], np.cumsum(metrics.qlike(rv_t, f_d ** (1 - w) * f_s ** w))]) for w in GRID])
    order = sorted(range(len(GRID)), key=lambda k: (abs(GRID[k] - 0.5), GRID[k]))
    wd = np.zeros(len(f_d))
    for k in range(len(f_d)):
        if m[k] >= MIN_CLOSED:
            mean = cum[:, m[k]] / m[k]
            best = order[0]
            for j in order[1:]:
                if mean[j] < mean[best] - 1e-12:
                    best = j
            wd[k] = GRID[best]
    return f_d ** (1 - wd) * f_s ** wd, wd


def reconcile(f: np.ndarray, comps: np.ndarray) -> tuple:
    """comps (n x K, >= 0) масштабируются так, что сумма строки = f. -> (компоненты, флаг «Σ <= 0 или нефинитна: поровну»)."""
    comps, f = np.asarray(comps, float), np.asarray(f, float)
    if (comps < 0).any() or not (f > 0).all():
        raise ValueError("согласование: отрицательная компонента или неположительная согласованная сумма")
    tot = comps.sum(axis=1)
    bad = ~np.isfinite(tot) | (tot <= 0)
    out = np.where(bad[:, None], f[:, None] / comps.shape[1], comps * (f / np.where(bad, 1.0, tot))[:, None])
    return out, bad


def run_r2(w: DC.DWorld, cfg: str, extra: int = 0) -> tuple:
    """R2x в мире w для компонентной конфигурации cfg схемы: -> (согласованная сумма, w по дням, σ̂²_Σ, A0, компоненты согласованные n x K, флаг fallback)."""
    a0 = DC.a0_forecast(w)
    s2, comps, fb = DC.run(w, cfg, a0)
    f, wd = blend(a0, s2, w.target, w.d, w.td, extra)
    names = list(comps)
    C = np.column_stack([np.where(fb, np.nan, comps[n]) for n in names])
    rec, bad = reconcile(f, np.where(np.isfinite(C), C, 0.0))
    return f, wd, s2, a0, rec, bad | fb


def sigma(w: DC.DWorld, cfg: str) -> np.ndarray:
    return run_r2(w, cfg)[0]
