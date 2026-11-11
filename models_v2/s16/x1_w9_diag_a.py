"""W9: блок A (разделение HAR-части и энкодера C: переносы W9A, клетки год × состояние) и блок B (ридж под остаток X1).
Перенос из scratchpad `w9/core.py` без изменения логики (лист кодера 11.11.2026)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models_v2.s16.x1_w9_diag_core import GRID, LAM, TRANSFERS, fit_kappa, fit_kappa2, qlike


def w9a(x: pd.DataFrame) -> dict:
    rv, lnM = x["rv"].to_numpy(float), np.log(x["M"].to_numpy(float))
    p1, p2 = (x["pi"] * x["DH"]).to_numpy(float), (x["pi"] * x["DG"]).to_numpy(float)
    piD = p1 + p2
    out = {}
    for t, a, b in TRANSFERS:
        A, B = x[a].to_numpy(bool), x[b].to_numpy(bool)
        kh, kg = fit_kappa2(rv, lnM, p1, p2, A)
        k0 = fit_kappa(rv, lnM, piD, A)
        qc = qlike(rv[B], np.exp(lnM[B] + kh * p1[B] + kg * p2[B])).mean()
        q0 = qlike(rv[B], np.exp(lnM[B] + k0 * piD[B])).mean()
        out[t] = dict(q_c=qc, q_0=q0, q_x1=x["lX1"].to_numpy(float)[B].mean(), k0=k0, kg={"kH": kh, "kG": kg}, nB=int(B.sum()))
    ep = {}
    for h in ("h1", "h2b22", "h2"):
        H = x[h].to_numpy(bool)
        ep[h] = fit_kappa2(rv, lnM, p1, p2, H)
    return dict(tr=out, ep=ep)


def cells(x: pd.DataFrame) -> pd.DataFrame:  # xtask: длинная — перенос без изменения логики, одна таблица клеток
    rv, lnM = x["rv"].to_numpy(float), np.log(x["M"].to_numpy(float))
    D, DH, DG = (x[c].to_numpy(float) for c in ("D", "DH", "DG"))
    rows = []
    for y in sorted(x["yr"].unique()):
        for st, sm in (("N", x["s"].to_numpy() == 0), ("E", x["s"].to_numpy() == 1), ("all", np.ones(len(x), bool))):
            m = (x["yr"].to_numpy() == y) & sm
            if m.sum() < 5:
                continue
            L1 = np.array([qlike(rv[m], np.exp(lnM[m] + k * D[m])).mean() for k in GRID])
            k1 = float(GRID[int(np.argmin(L1))])
            kh, kg = fit_kappa2(rv, lnM, DH, DG, m)
            c2 = qlike(rv[m], np.exp(lnM[m] + kh * DH[m] + kg * DG[m])).mean()
            rows.append(dict(period=int(y), state=st, n=int(m.sum()), corr_D_DH=float(np.corrcoef(D[m], DH[m])[0, 1]),
                             corr_D_DG=float(np.corrcoef(D[m], DG[m])[0, 1]), sd_DH=float(DH[m].std(ddof=1)), sd_DG=float(DG[m].std(ddof=1)),
                             mean_absDG_share=float(np.mean(np.abs(DG[m]) / (np.abs(DG[m]) + np.abs(DH[m]) + 1e-300))),
                             Q_X1=float(x["lX1"].to_numpy(float)[m].mean()), Q_M=float(L1[0]), c1=float(L1.min()), k1=k1, c2=float(c2), kH=kh, kG=kg,
                             mean_pi=float(x["pi"].to_numpy(float)[m].mean())))
    return pd.DataFrame(rows)


# ---------- W9B ----------

def fit_ridge(u, Z, lam=LAM):
    n, p = Z.shape

    def f(th):
        g = th[0] + Z @ th[1:]
        e = np.exp(u - g)
        val = np.mean(e + g) + lam * np.sum(th[1:] ** 2)
        dg = (1.0 - e) / n
        grad = np.concatenate([[dg.sum()], Z.T @ dg + 2 * lam * th[1:]])
        return val, grad

    th0 = np.zeros(p + 1)
    th0[0] = np.log(np.mean(np.exp(u)))
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-10))
    return r.x


def ridge_transfer(x: pd.DataFrame, cols: list, roll: int = 0) -> dict:
    rv, X1 = x["rv"].to_numpy(float), x["F"].to_numpy(float)
    u = np.log(rv / X1)
    X = x[cols].to_numpy(float)
    if roll:
        X = np.roll(X, roll, axis=0)
    out = {}
    for t, a, b in TRANSFERS:
        A, B = x[a].to_numpy(bool), x[b].to_numpy(bool)
        mu, sd = X[A].mean(0), X[A].std(0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (X - mu) / sd
        th = fit_ridge(u[A], Z[A])
        g = np.clip(th[0] + Z[B] @ th[1:], -1, 1)
        c0 = np.clip(np.log(np.mean(np.exp(u[A]))), -1, 1)
        qc = qlike(rv[B], X1[B] * np.exp(g)).mean()
        q0 = qlike(rv[B], X1[B] * np.exp(c0)).mean()
        out[t] = dict(q_c=qc, q_0=q0, q_x1=x["lX1"].to_numpy(float)[B].mean(), k0=float(c0), kg={"th0": float(th[0]), "norm_th": float(np.linalg.norm(th[1:]))}, nB=int(B.sum()))
    return out


def verdict_ridge(tr: dict) -> dict:
    d0 = {t: (tr[t]["q_c"] / tr[t]["q_0"] - 1) * 100 for t in tr}
    dx = {t: (tr[t]["q_c"] / tr[t]["q_x1"] - 1) * 100 for t in tr}
    P = dict(P1=d0["T1"] < 0 and d0["T2"] < 0, P2=dx["T1"] < 0 and dx["T2"] < 0, P3=d0["T3"] < 0 and d0["T4"] < 0, P5=dx["T3"] <= 0 and dx["T4"] <= 0)
    P["all"] = all(P.values())
    return dict(P=P, d0=d0, dx=dx)
