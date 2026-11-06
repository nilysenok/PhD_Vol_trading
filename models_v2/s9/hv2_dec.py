"""Ж-214, лист 3, V7: декомпозиции RV (схемы S — знак, J — скачок/непрерывная, T — время суток), каждая — самостоятельная; компоненты разных схем не суммируются.
Компонента x_k (%², строка t, известна к 18:50 t): признаки [ln x_d, ln x_w, ln x_m] (x_w, x_m — средние уровней за 5/22 дня сегмента, включая t; для J — ln(1+·)), цель ln x_{k,t+1},
σ̂²_k = exp(f + s²/2), OLS по парам t+1 <= d, один сегмент, расширяющееся окно (У10–У14 GATES_V2). σ̂²_Σ = Σ_k σ̂²_k > 0. Нефинитный признак строки d (нулевая компонента, окно не набрано)
или < k+2 пар → прогноз A0 на весь день (`n_fallback`). C := RV − J (BV в сумму не входит)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import daily as D, har as H
from models_v2.s9 import hv2_a0 as A0, hv2_world as W

KEYS = ("c", "j", "rs_pos", "rs_neg", "rv_open", "rv_mid", "rv_close")
CONFIGS = {"S1": "S", "S3": "S", "J1": "J", "T1": "T", "T3": "T"}
COMPS = {"S1": ("rs_pos", "rs_neg"), "S3": ("rv",), "J1": ("c", "j"), "T1": ("rv_open", "rv_mid", "rv_close"), "T3": ("rv_open", "rv_mid", "rv_close")}


@dataclass
class DWorld(W.World):
    comp: dict | None = None     # имя -> уровень компоненты (%²), строка t, выровнено на p.dates


def build(w: W.World, m10: pd.DataFrame, keys=KEYS) -> DWorld:
    m = m10.reindex(pd.DatetimeIndex(w.p.dates))
    return DWorld(**{f.name: getattr(w, f.name) for f in dataclasses.fields(W.World)}, comp={k: m[k].to_numpy(float) for k in keys})


def corrupt(w: DWorld, i0: int, seed: int = 7) -> DWorld:
    """Главный тест обрезки: строки > i0 всех величин (в т. ч. индикатор J > 0 и нули компонент) испорчены; строки <= i0 без изменений."""
    rng, n = np.random.default_rng(seed + 1), len(w.p.rv)
    out = W.corrupt(w, i0, seed)
    late = np.arange(n) > i0
    comp = {}
    for k, x in w.comp.items():
        v = np.where(late, x * np.exp(rng.normal(0, 1.0, n)), x)
        comp[k] = np.where(late & (rng.random(n) < 0.4), 0.0, v)    # нули: ломают и индикатор скачка, и логарифм компоненты
    return dataclasses.replace(out, comp=comp)


def trunc_diff(run_fn, w: DWorld, i0: int) -> float:
    """σ̂² на дни <= i0 из исходного и испорченного мира (run_fn(world) -> σ̂² сетки или кортеж)."""
    def f(x):
        o = run_fn(W.sub(x, i0))
        return np.asarray(o[0] if isinstance(o, tuple) else o, float)
    a, b = f(w), f(corrupt(w, i0))
    if (np.isnan(a) != np.isnan(b)).any():
        return float("inf")                      # маска определённости прогноза зависит от будущего — тоже утечка
    return float(np.nanmax(np.abs(a / b - 1))) if np.isfinite(a).any() else 0.0


def wmean(x: np.ndarray, seg: np.ndarray, k: int) -> np.ndarray:
    """Среднее уровней за k строк сегмента, включая t (NaN, пока окно не набрано или есть NaN)."""
    out = np.full(len(x), np.nan)
    for s in np.unique(seg):
        ix = np.flatnonzero(seg == s)
        if len(ix) >= k:
            out[ix[k - 1:]] = np.lib.stride_tricks.sliding_window_view(x[ix], k).mean(axis=1)
    return out


def own(x: np.ndarray, seg: np.ndarray, log1p: bool = False) -> np.ndarray:
    f = np.log1p if log1p else np.log
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([f(x), f(wmean(x, seg, 5)), f(wmean(x, seg, 22))])


def lnx(x: np.ndarray, log1p: bool = False) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(x) if log1p else np.log(x)


def forecast(w: W.World, F: np.ndarray, y: np.ndarray, j_level: bool = False) -> np.ndarray:
    """σ̂² (или Ĵ при j_level) на сетке по логарифмической регрессии y_{t+1} на F_t; NaN, если признаки строки d нефинитны или мало пар."""
    out = np.full(len(w.el), np.nan)
    for a, i in enumerate(w.el):
        if not np.isfinite(F[i]).all():
            continue
        idx = W.pairs_idx(w.p.seg, F, int(i))
        idx = idx[np.isfinite(y[idx + 1])]
        if len(idx) < F.shape[1] + 2:
            continue
        b, v, _ = H.ols(F[idx], y[idx + 1])
        f = float(b[0] + np.einsum("j,j->", b[1:], F[i]) + v / 2)
        out[a] = max(np.expm1(f), 0.0) if j_level else float(np.exp(f))
    return out


def spec(w: DWorld, cfg: str) -> list:
    """Список уравнений конфигурации: (имя, матрица признаков, цель, j_level). J регрессор — ln(1+J_d) панели A0 (p.X[:, 3])."""
    seg, X, c = w.p.seg, w.p.X, w.comp
    J = X[:, 3:4]
    if cfg == "S3":
        return [("rv", np.column_stack([lnx(c["rs_pos"]), lnx(c["rs_neg"]), X[:, 1], X[:, 2], J]), w.p.lnrv, False)]
    if cfg == "J1":
        return [("c", np.column_stack([own(c["c"], seg), J]), lnx(c["c"]), False),
                ("j", own(c["j"], seg, True), lnx(c["j"], True), True)]
    names = COMPS[cfg]
    edge = {"rv_open": "rv_close", "rv_mid": "rv_close", "rv_close": "rv_open"}
    eqs = []
    for k in names:
        F = np.column_stack([own(c[k], seg), J] + ([lnx(c[edge[k]])] if cfg == "T3" else []))
        eqs.append((k, F, lnx(c[k]), False))
    return eqs


def a0_forecast(w: W.World) -> np.ndarray:
    return A0.harj_series(w.p, w.el)["sigma2_b"].to_numpy(float)


def run(w: DWorld, cfg: str, a0: np.ndarray | None = None) -> tuple:
    """-> (σ̂² сетки с подменой на A0, {компонента: σ̂²_k}, маска fallback). Прогнозы компонент — NaN в днях fallback не затираются (для таблицы компонент)."""
    comps = {n: forecast(w, F, y, jl) for n, F, y, jl in spec(w, cfg)}
    tot = np.sum([comps[n] for n in comps], axis=0)
    fb = ~np.isfinite(tot) | (tot <= 0)
    a0 = a0_forecast(w) if a0 is None else a0
    return np.where(fb, a0, tot), comps, fb


def sigma(w: DWorld, cfg: str) -> np.ndarray:
    return run(w, cfg)[0]


def identities(s2: np.ndarray, comps: dict, fb: np.ndarray) -> dict:
    """Положительность и сумма на днях без fallback: min компонент, max |Σ − σ̂²|/σ̂², n_fallback."""
    ok = ~fb
    tot = np.sum([comps[n][ok] for n in comps], axis=0)
    return {"min_component": float(min(np.min(comps[n][ok]) for n in comps)), "max_rel_sum_err": float(np.max(np.abs(tot / s2[ok] - 1))),
            "min_sigma2": float(np.min(s2)), "n_fallback": int(fb.sum()), "n": len(s2)}
