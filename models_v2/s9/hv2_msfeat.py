"""Ж-214, лист 4 (V9, V10): мир с M10-величинами (DWorld с расширенным `comp`), PIT-базы на РАСШИРЕННЫХ строках (A0, B* = H2, компоненты T3: обучающим парам ML нужны прогнозы базы
и до начала сетки) и признаки «MS» по группам Q, S, T, M (У24 GATES_V2). Все признаки строки t известны к 18:50 t; окна — внутри сегмента, включая t.
Базы пересчитываются кодом (не читаются из рядов): Д5 — смена кода/входа даёт промах кэша; сверка с хранимыми рядами на сетке — тест."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_a0 as A0, hv2_dec as DC, hv2_harq2 as HR

MIN_PAIRS = 252
KEYS = DC.KEYS + ("corr", "beta", "cov", "cov_pp", "cov_nn", "cov_mix", "rv_s_sync", "rv_m_sync", "rsm_pos", "rsm_neg")
SPEC_H2 = {"q": "rq", "react": ("J",)}
GROUPS = ("Q", "S", "T", "M")
CORE = ("ln_rv_d", "ln_rv_w", "ln_rv_m", "ln1p_J", "ln_base")


def build_world(w, m10: pd.DataFrame, imoex_bars: pd.DataFrame) -> DC.DWorld:
    """DWorld: M10-компоненты (в т. ч. синхронные с IMOEX) и дневной ряд IMOEX (rv, rv_w, rv_m; собственные дни индекса) на датах панели."""
    dw = DC.build(w, m10, KEYS)
    ix = D.build_daily(imoex_bars, index_days=None)[0].reindex(pd.DatetimeIndex(w.p.dates))
    comp = {**dw.comp, "imo_rv": ix["rv"].to_numpy(float), "imo_w": ix["rv_w"].to_numpy(float), "imo_m": ix["rv_m"].to_numpy(float)}
    return dataclasses.replace(dw, comp=comp)


def ln(x) -> np.ndarray:
    x = np.asarray(x, float)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def lnr(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = (a > 0) & (b > 0)
    return np.where(ok, np.log(np.where(ok, a, 1.0) / np.where(ok, b, 1.0)), np.nan)


def fin(x) -> np.ndarray:
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), x, np.nan)


def ext_rows(w, design: np.ndarray, upto: int | None = None, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    """Строки i, где дизайн конечен и пар (t, t+1), t+1 <= i, одного сегмента с конечным дизайном — не меньше min_pairs; i <= upto (по умолчанию max сетки)."""
    seg, n = w.p.seg, len(w.p.rv)
    ok = np.isfinite(design).all(axis=1)
    valid = np.zeros(n, bool)
    valid[:-1] = (seg[:-1] == seg[1:]) & ok[:-1]
    npairs = np.concatenate([[0], np.cumsum(valid)[:-1]])
    rows = np.flatnonzero(ok & (npairs >= min_pairs))
    return rows[(rows < n - 1) & (rows <= (int(w.el.max()) if upto is None else upto))]


def _fill(n: int, rows: np.ndarray, v) -> np.ndarray:
    a = np.full(n, np.nan)
    a[rows] = v
    return a


def base_a0(w, rows: np.ndarray) -> np.ndarray:
    return _fill(len(w.p.rv), rows, A0.harj_series(w.p, rows)["sigma2_b"].to_numpy(float))


def base_h2(w, rows: np.ndarray) -> np.ndarray:
    return _fill(len(w.p.rv), rows, HR.run(dataclasses.replace(w, el=rows), SPEC_H2)[0])


def base_t3(dw, rows: np.ndarray, a0: np.ndarray) -> dict:
    """T3 на строках rows: σ̂² (с подменой на A0 в днях fallback), прогнозы компонент, маска fallback (1.0)."""
    s2, comps, fb = DC.run(dataclasses.replace(dw, el=rows), "T3", a0[rows])
    n = len(dw.p.rv)
    return {"s2": _fill(n, rows, s2), "fb": _fill(n, rows, fb.astype(float)), "comps": {k: _fill(n, rows, v) for k, v in comps.items()}}


def bases(dw, upto: int | None = None, need=("A0", "H2", "T3"), min_pairs: int = MIN_PAIRS) -> dict:
    """{'A0': σ̂²_b, 'H2': σ̂², 'T3': {...}} на расширенных строках (NaN вне них); считаются только нужные (T3 требует A0). PIT: прогноз строки i — по парам t+1 <= i."""
    out = {}
    if "A0" in need or "T3" in need:
        rows_a = ext_rows(dw, dw.p.X, upto, min_pairs)
        out["A0"] = base_a0(dw, rows_a)
        if "T3" in need:
            out["T3"] = base_t3(dw, rows_a, out["A0"])
    if "H2" in need:
        out["H2"] = base_h2(dw, ext_rows(dw, HR.design(dw, SPEC_H2)[0], upto, min_pairs))
    return out


def wm(dw, x, k: int) -> np.ndarray:
    return DC.wmean(np.asarray(x, float), dw.p.seg, k)


def group_cols(dw, g: str) -> list:
    """Столбцы группы g: [(имя, массив n)]. Q — качество, S — знак и скачок, T — время суток, M — рынок (синхронные бары)."""
    c, rv = dw.comp, dw.p.rv
    with np.errstate(divide="ignore", invalid="ignore"):
        if g == "Q":
            return [("q_ln_rel", ln(dw.rel)), ("q_lnrq_minus_lnrv", 0.5 * ln(dw.rq) - ln(rv))]
        if g == "S":
            return [("s_lnnegpos_d", lnr(c["rs_neg"], c["rs_pos"])), ("s_lnnegpos_w", lnr(wm(dw, c["rs_neg"], 5), wm(dw, c["rs_pos"], 5))),
                    ("s_jshare_d", c["j"] / rv), ("s_jshare_w", wm(dw, c["j"], 5) / wm(dw, rv, 5))]
        if g == "T":
            return [("t_open_mid_d", lnr(c["rv_open"], c["rv_mid"])), ("t_close_mid_d", lnr(c["rv_close"], c["rv_mid"])),
                    ("t_open_mid_w", lnr(wm(dw, c["rv_open"], 5), wm(dw, c["rv_mid"], 5))), ("t_close_mid_w", lnr(wm(dw, c["rv_close"], 5), wm(dw, c["rv_mid"], 5)))]
        if g == "M":
            den = c["cov_pp"] + c["cov_nn"] + np.abs(c["cov_mix"])
            idio = 1.0 - c["beta"] ** 2 * c["rv_m_sync"] / c["rv_s_sync"]
            return [("m_corr", c["corr"]), ("m_beta", c["beta"]), ("m_semicov_asym", (c["cov_pp"] - c["cov_nn"]) / den), ("m_idio_share", idio)]
    raise ValueError("группа %r" % g)


def feature_matrix(dw, ln_base: np.ndarray, groups=GROUPS) -> tuple:
    """(X n×k, имена): ядро [ln RV_d, ln RV_w, ln RV_m, ln(1+J), ln база] + группы. Первые 4 столбца — ворота пар (конечны), остальные могут быть NaN (деревья)."""
    X = dw.p.X
    cols = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], np.asarray(ln_base, float)]
    names = list(CORE)
    for g in groups:
        for nme, v in group_cols(dw, g):
            cols.append(fin(v))
            names.append(nme)
    return np.column_stack(cols), names
