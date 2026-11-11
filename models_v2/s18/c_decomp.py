"""S-C-DATA: проекционное разложение RV SBER сессии s на common / specific / uncovered (Ж-254 п. 8). Прогнозов нет.

На покрытых барах (фактор определён): r = P r + (I − P) r, P — МНК-проекция на пространство 1–2 факторов В ЭТОЙ сессии (без свободного члена).
rv_common = ‖P r‖², rv_specific = ‖(I − P) r‖², rv_uncovered = Σ r² по барам без фактора; их сумма = RV целевого определения.
Коэффициенты проекции сессии s известны только после её конца: это метка для t⁺, не признак дня d−1 и не вход прогноза.
Вариант «PIT-нагрузки»: коэффициент проекции b оценён по прошлому окну (те же нагрузки, σ, фактор, SBER прошлых сессий) и применён к
фактору сессии s без пересчёта внутри неё; перекрёстный член 2⟨P_b r, r − P_b r⟩ не исчезает — он сохраняется отдельно.
Единицы — %² (×1e4), как `daily.rv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import c_factor as CF
from models_v2.s18.c_data import Inputs

SC = 1e4
MIN_COV = 10        # покрытых баров сессии для проекции; меньше — вся RV в uncovered
MIN_PIT = 200       # покрытых баров окна для коэффициента PIT-варианта


def parts(r: np.ndarray, F: np.ndarray, b: np.ndarray | None = None) -> dict:
    """r (n,), F (n,k) -> common, specific, unc, cross (в r²), n_cov. b=None — МНК по сессии; иначе заданный коэффициент (без пересчёта)."""
    cov = np.isfinite(F).all(1)
    out = dict(common=0.0, specific=0.0, unc=float(np.square(r[~cov]).sum()), cross=0.0, n_cov=int(cov.sum()))
    if out["n_cov"] < MIN_COV or (b is not None and not np.isfinite(b).all()):
        out["unc"] = float(np.square(r).sum())
        out["n_cov"] = 0 if out["n_cov"] < MIN_COV else out["n_cov"]
        return out
    Fc, rc = F[cov], r[cov]
    b = np.linalg.lstsq(Fc, rc, rcond=None)[0] if b is None else b
    Pr = Fc @ b
    e = rc - Pr
    out.update(common=float(Pr @ Pr), specific=float(e @ e), cross=float(2 * Pr @ e))
    return out


def pit_coef(inp: Inputs, lo: int, hi: int, sig, Wk) -> np.ndarray | None:
    """Коэффициент проекции SBER на фактор по барам окна [lo, hi) (только прошлое); None — мало покрытых баров."""
    Fw = CF.scores(inp.R[lo:hi], sig, Wk)
    cov = np.isfinite(Fw).all(1) & np.isfinite(inp.r[lo:hi])
    if cov.sum() < MIN_PIT:
        return None
    return np.linalg.lstsq(Fw[cov], inp.r[lo:hi][cov], rcond=None)[0]


def session_rows(inp: Inputs, s: int) -> list:
    """Строки сессии s для k = 1 и 2 факторов."""
    a, e = int(inp.off[s]), int(inp.off[s + 1])
    r, rv = inp.r[a:e], float(inp.rv[s])
    lo, hi, nh = CF.window(inp.off, s)
    ld = CF.loadings(inp.R[lo:hi], 2) if nh >= CF.MIN_HIST else None
    rows = []
    for k in (1, 2):
        row = dict(d=inp.days[s], k_factors=k, rv=rv, n_bars=int(inp.nbars[s]), n_sber_bars=e - a, flag_nohist=nh < CF.MIN_HIST,
                   flag_noload=ld is None and nh >= CF.MIN_HIST)
        F = np.full((e - a, k), np.nan) if ld is None else CF.scores(inp.R[a:e], ld[0], ld[1][:, :k])
        bp = None if ld is None else pit_coef(inp, lo, hi, ld[0], ld[1][:, :k])
        p, q = parts(r, F), parts(r, F, bp if bp is not None else np.full(k, np.nan))
        row.update(rv_common=SC * p["common"], rv_specific=SC * p["specific"], rv_uncovered=SC * p["unc"], cross=SC * p["cross"],
                   n_cov=p["n_cov"], flag_fewcov=bool(ld is not None and p["n_cov"] == 0),
                   n_pap_med=float(np.median(np.isfinite(inp.R[a:e][np.isfinite(F).all(1)]).sum(1))) if p["n_cov"] else 0.0,
                   rv_common_pit=SC * q["common"], rv_specific_pit=SC * q["specific"], cross_pit=SC * q["cross"], flag_nopit=bp is None)
        row["ident_err"] = row["rv_common"] + row["rv_specific"] + row["rv_uncovered"] - rv
        row["ident_err_pit"] = row["rv_common_pit"] + row["rv_specific_pit"] + row["cross_pit"] + SC * q["unc"] - rv
        rows.append(row)
    return rows


def run(inp: Inputs, sessions=None) -> pd.DataFrame:
    """Таблица по сессиям (по две строки: k = 1, 2)."""
    idx = range(len(inp.days)) if sessions is None else sessions
    return pd.DataFrame([row for s in idx for row in session_rows(inp, s)])
