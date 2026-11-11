"""W9 core: данные окна M, разложение C, переносы h1 <-> h2 (объявлено в W9_DECL.md до счёта). Только чтение сохранённых рядов.
Перенос из scratchpad `w9/core.py` без изменения логики (лист кодера 11.11.2026); `ROOT` — от файла. Часть W9A / ridge — в `x1_w9_diag_a.py`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_groups as DG
from models_v2.s16 import x1_w3_core as W3

ROOT = Path(__file__).resolve().parents[2]

GRID = np.round(np.arange(0, 31) * 0.05, 2)          # 0 … 1,5
MIN_G = 30
MIN_PAST = 60
LAM = 0.01
SHIFTS = [252 + 100 * (j - 1) for j in range(1, 20)]
N_PERM = 500
SIGMA12 = ["a_PV", "a_PN", "a_PQ", "a_VN", "a_VQ", "a_NQ", "m5_a_PV", "m5_a_PN", "m5_a_PQ", "m5_a_VN", "m5_a_VQ", "m5_a_NQ"]
S9 = ["s%d" % i for i in range(1, 10)]
SRC_EXTRA = {"cfg8_a": "data/fwd/models/cfg8.parquet", "cfg8_n": "data/fwd/models/new/cfg8.parquet",
             "S_ext": "data/regen/harj_x1/W4/feat/S_ext.parquet", "SIGMA_ext": "data/regen/harj_x1/W4/feat/SIGMA_ext.parquet",
             "z2026_s0": "data/fwd/models/z_tcn2026_s0.npz"}


def qlike(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def z_loader(year: int, seed: int):
    if year == 2026:
        a = np.load(ROOT / "data/fwd/models" / ("z_tcn2026_s%d.npz" % seed))
    else:
        a = np.load(ROOT / "data/regen/harj_hybrid_v7/stageA" / ("tcn_w5_%d_s%d.npz" % (year, seed)))
    return pd.DatetimeIndex(a["days"].astype("datetime64[ns]")), np.asarray(a["z"], float)


def z_overlap_check() -> float:
    """max |z fwd 2026 − z stageA 2026| на общих днях (все seeds)."""
    m = 0.0
    for s in range(5):
        d1, z1 = z_loader(2026, s)
        a = np.load(ROOT / "data/regen/harj_hybrid_v7/stageA" / ("tcn_w5_2026_s%d.npz" % s))
        d2, z2 = pd.DatetimeIndex(a["days"].astype("datetime64[ns]")), np.asarray(a["z"], float)
        common = d2.intersection(d1)
        m = max(m, float(np.abs(z1[d1.get_indexer(common)] - z2[d2.get_indexer(common)]).max()))
    return m


def definition_checks(x: pd.DataFrame) -> dict:
    """Сверки определения (Д-1): F_cfg8 = C; ln F8 = ln H + G; ell = ln(M/H); H = K1; D = D_H + D_G; ln X1 = ln M + π·D.
    Добавляет колонки D, DH, DG (D_G = G − w·ell, а не G — Д-1)."""
    chk = {}
    chk["C_eq_F8"] = float(np.max(np.abs(x["F8"] / x["C"] - 1)))
    # колонка G = полный лог-сдвиг cfg8 над H (якорь w·ℓ + Σπ_k g_k, среднее по seeds): ln F8 = ln H + G (Д-1)
    chk["lnF8_eq_lnH_G"] = float(np.max(np.abs(np.log(x["F8"]) - (np.log(x["H"]) + x["G"]))))
    chk["ell_eq_ln_M_H"] = float(np.max(np.abs(x["ell"] - np.log(x["M"] / x["H"]))))
    chk["H_eq_K1"] = float(np.max(np.abs(x["H"] / x["K1"] - 1)))
    x["D"] = np.log(x["C"] / x["M"])
    x["DH"] = -(1.0 - x["w"]) * x["ell"]
    x["DG"] = x["G"] - x["w"] * x["ell"]
    chk["D_eq_DH_DG"] = float(np.max(np.abs(x["D"] - x["DH"] - x["DG"])))
    chk["lnX1_eq"] = float(np.max(np.abs(np.log(x["F"]) - (np.log(x["M"]) + x["pi"] * x["D"]))))
    for k, v in chk.items():
        tol = 1e-8 if k != "H_eq_K1" else 1e-10
        if not v <= tol:
            raise SystemExit("сверка %s: %g" % (k, v))
    return chk


def load(with_z: bool = True) -> tuple[pd.DataFrame, dict]:
    x = W3.load()
    ca = pd.read_parquet(ROOT / SRC_EXTRA["cfg8_a"])
    cn = pd.read_parquet(ROOT / SRC_EXTRA["cfg8_n"])
    c8 = pd.concat([ca, cn], ignore_index=True)
    c8["d"] = pd.to_datetime(c8["d"])
    if c8["d"].duplicated().any():
        raise SystemExit("cfg8: дубль d")
    x = x.merge(c8[["d", "H", "ell", "w", "G", "F", "pi0", "pi1", "pi2"]].rename(columns={"F": "F8"}), on="d", how="left")
    if x[["H", "ell", "w", "G", "F8"]].isna().any().any():
        raise SystemExit("cfg8: нет строк окна")
    chk = definition_checks(x)
    S = pd.read_parquet(ROOT / SRC_EXTRA["S_ext"])
    Sg = pd.read_parquet(ROOT / SRC_EXTRA["SIGMA_ext"])
    for t in (S, Sg):
        t["d"] = pd.to_datetime(t["d"])
    x = x.merge(S[["d"] + S9], on="d", how="left").merge(Sg[["d"] + SIGMA12], on="d", how="left")
    if x[S9 + SIGMA12].isna().any().any():
        raise SystemExit("S/Σ: пропуски на датах окна: %d" % int(x[S9 + SIGMA12].isna().any(axis=1).sum()))
    if with_z:
        x["zinst"] = DG.z_distance(pd.DatetimeIndex(x["d"]), z_loader)
        chk["z_fwd_vs_stageA_2026"] = z_overlap_check()
        chk["zinst_nan"] = float(np.isnan(x["zinst"]).sum())
    yr = x["t_plus"].dt.year.to_numpy()
    x["yr"] = yr
    x["h1"] = yr <= 2021
    x["h2"] = yr >= 2022
    x["h2b22"] = (yr >= 2023)
    x["lX1"] = qlike(x["rv"], x["F"])
    return x, chk


# ---------- PIT-терцили ----------

def pit_terciles(v: np.ndarray, min_past: int = MIN_PAST) -> np.ndarray:
    """Группа дня i по порогам (1/3, 2/3) значений дней строго до i (не NaN, ≥ min_past); иначе −1."""
    v = np.asarray(v, float)
    out = np.full(len(v), -1, int)
    for i in range(len(v)):
        past = v[:i]
        past = past[np.isfinite(past)]
        if len(past) < min_past or not np.isfinite(v[i]):
            continue
        q1, q2 = np.quantile(past, [1 / 3, 2 / 3])
        out[i] = 0 if v[i] <= q1 else (1 if v[i] <= q2 else 2)
    return out


# ---------- подгонка κ по решётке ----------

def fit_kappa(rv, lnM, piD, mask) -> float:
    """argmin_κ mean QLIKE(rv, exp(lnM + κ·piD)) на mask; ничья — меньшее κ."""
    r, a, b = rv[mask], lnM[mask], piD[mask]
    L = np.array([qlike(r, np.exp(a + k * b)).mean() for k in GRID])
    return float(GRID[int(np.argmin(L))])


def fit_kappa2(rv, lnM, p1, p2, mask) -> tuple:
    r, a, b, c = rv[mask], lnM[mask], p1[mask], p2[mask]
    best, arg = np.inf, (0.0, 0.0)
    for kh in GRID:
        base = a + kh * b
        L = np.array([qlike(r, np.exp(base + kg * c)).mean() for kg in GRID])
        j = int(np.argmin(L))
        if L[j] < best - 1e-15:
            best, arg = L[j], (float(kh), float(GRID[j]))
    return arg


TRANSFERS = (("T1", "h1", "h2"), ("T2", "h2", "h1"), ("T3", "h1", "h2b22"), ("T4", "h2b22", "h1"))


def group_transfer(x: pd.DataFrame, g: np.ndarray, lnM, piD) -> dict:
    """Перенос группового κ: -> {T: dict(q_c, q_0, q_x1, k0, kg)}."""
    rv = x["rv"].to_numpy(float)
    out = {}
    for t, a, b in TRANSFERS:
        A, B = x[a].to_numpy(bool), x[b].to_numpy(bool)
        k0 = fit_kappa(rv, lnM, piD, A)
        kg = {}
        for gg in sorted(set(g[g >= 0])):
            m = A & (g == gg)
            if m.sum() >= MIN_G:
                kg[int(gg)] = fit_kappa(rv, lnM, piD, m)
        kap = np.array([kg.get(int(gg), k0) for gg in g])
        qc = qlike(rv[B], np.exp(lnM[B] + kap[B] * piD[B])).mean()
        q0 = qlike(rv[B], np.exp(lnM[B] + k0 * piD[B])).mean()
        qx = x["lX1"].to_numpy(float)[B].mean()
        out[t] = dict(q_c=qc, q_0=q0, q_x1=qx, k0=k0, kg=kg, nB=int(B.sum()))
    return out


def expost_groups(x, g, lnM, piD) -> dict:
    rv = x["rv"].to_numpy(float)
    res = {}
    for h in ("h1", "h2b22"):
        H = x[h].to_numpy(bool)
        res[h] = {int(gg): (fit_kappa(rv, lnM, piD, H & (g == gg)), int((H & (g == gg)).sum())) for gg in sorted(set(g[g >= 0]))}
    return res


def same_order(e1: dict, e2: dict) -> bool:
    keys = [k for k in e1 if k in e2 and e1[k][1] >= MIN_G and e2[k][1] >= MIN_G]
    if len(keys) < 2:
        return False
    for i in keys:
        for j in keys:
            if i < j and np.sign(e1[i][0] - e1[j][0]) != np.sign(e2[i][0] - e2[j][0]):
                return False
    return True


def verdict_group(tr: dict, ep: dict) -> dict:
    d0 = {t: (tr[t]["q_c"] / tr[t]["q_0"] - 1) * 100 for t in tr}
    dx = {t: (tr[t]["q_c"] / tr[t]["q_x1"] - 1) * 100 for t in tr}
    P = dict(P1=d0["T1"] < 0 and d0["T2"] < 0, P2=dx["T1"] < 0 and dx["T2"] < 0, P3=d0["T3"] < 0 and d0["T4"] < 0,
             P4=same_order(ep["h1"], ep["h2b22"]), P5=dx["T3"] <= 0 and dx["T4"] <= 0)
    P["all"] = all(P.values())
    return dict(P=P, d0=d0, dx=dx)
