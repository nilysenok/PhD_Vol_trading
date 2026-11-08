"""V4-DX, G26, пункты 1–6 Ж-225 п. 7: диагностика недостатка информации по годам — ЗАДНИМ ЧИСЛОМ, описательно, в прогноз и выбор не идёт (модуль `v4_dx_*` не импортируется путями прогноза).
Срез = год цели (t⁺) или окно × группа (M16 — 16 бумаг без SBER, SBER отдельно) × M ∈ {KG = √(K2·K3), K2, K3}; H = K1. Строки — общие даты H ∩ K2 ∩ K3 (`v4_io.aligned`).
D = QLIKE(H) − QLIKE(M): минус — HAR-J лучше. Вес HAR-J — в лучшей геометрической смеси F = H^w·M^(1−w) задним числом на пуле строк группы; 90 % интервал — блочный бутстрап дат (блок 20, 300 выборок, seed 42, `stats_d.block_idx`);
w ∈ [−1; 2] (интервал не упирается в границу [0; 1]); «ML охватывает HAR-J» — нижняя граница интервала ≤ 0, «HAR-J охватывает ML» — верхняя ≥ 1.
O1 (лучший постоянный вес, арифм./геом., [0; 1]), O2 = clip(RV, min(H, M), max(H, M)), O3 = b·O2 с b = mean(RV/O2) — определения `v4_oracle` (G7), здесь воспроизведены локально (oracle в DX не импортируется)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models_v2.s5.d import stats_d as SD
from models_v2.s11 import v4_io as IO

MS = {"KG": lambda a: np.sqrt(a["k2"].to_numpy(float) * a["k3"].to_numpy(float)), "K2": lambda a: a["k2"].to_numpy(float), "K3": lambda a: a["k3"].to_numpy(float)}
W_LO, W_HI, B_DRAWS, ITER = -1.0, 2.0, 300, 30
GRID = np.linspace(0.0, 1.0, 101)


def ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def panel_rows(tickers) -> pd.DataFrame:
    """Общие даты H ∩ K2 ∩ K3: ticker, date, t_plus, rv, h, k2, k3."""
    fr = []
    for t in tickers:
        a = IO.aligned(t)
        a.insert(0, "ticker", t)
        fr.append(a.drop(columns=["d2"]))
    return pd.concat(fr, ignore_index=True)


def solve_w(rv_e, ell, cnt, lo=W_LO, hi=W_HI) -> np.ndarray:
    """Вес на M в лучшей геометрической смеси для каждой строки cnt (B, R) — кратности строк в выборке; rv_e = RV/H, ell = ln(M/H). Бисекция по производной QLIKE (выпуклая по w)."""
    a, b = np.full(len(cnt), lo), np.full(len(cnt), hi)
    for _ in range(ITER):
        m = 0.5 * (a + b)
        g = (cnt * ell * (1.0 - rv_e * np.exp(-m[:, None] * ell))).sum(1)
        a, b = np.where(g > 0, a, m), np.where(g > 0, m, b)
    return 0.5 * (a + b)


def har_weight(rv, h, m, dates) -> dict:
    """Вес HAR-J (1 − w_M) на пуле строк и 90 % интервал блочного бутстрапа по датам."""
    ell, rv_e = np.log(m / h), rv / h
    _, code = np.unique(dates, return_inverse=True)
    nd = code.max() + 1
    w0 = float(solve_w(rv_e, ell, np.ones((1, len(rv))))[0])
    idx = SD.block_idx(nd, B_=B_DRAWS)
    cnt = np.stack([np.bincount(i, minlength=nd) for i in idx])[:, code].astype(float)
    ws = 1.0 - solve_w(rv_e, ell, cnt)
    lo, hi = np.quantile(ws, [0.05, 0.95])
    return {"w_har": 1.0 - w0, "w_har_lo90": float(lo), "w_har_hi90": float(hi), "w_har01": float(np.clip(1.0 - w0, 0.0, 1.0)), "ml_encompasses_har": bool(lo <= 0.0), "har_encompasses_ml": bool(hi >= 1.0)}


def o_values(rv, h, m) -> dict:
    """O1, O2, O3 одной бумаги (QLIKE); определения `v4_oracle`."""
    q = lambda f: float(np.mean(ql(rv, f)))
    mix = lambda w, geom: np.exp((1 - w) * np.log(h) + w * np.log(m)) if geom else (1 - w) * h + w * m
    best = min(q(h), q(m))
    for geom in (False, True):
        v = np.array([q(mix(w, geom)) for w in GRID])
        i = int(v.argmin())
        r = minimize_scalar(lambda w: q(mix(w, geom)), bounds=(GRID[max(i - 1, 0)], GRID[min(i + 1, 100)]), method="bounded", options={"xatol": 1e-10})
        best = min(best, float(v[i]), float(r.fun))
    x2 = np.clip(rv, np.minimum(h, m), np.maximum(h, m))
    b = float(np.mean(rv / x2))
    return {"o1": best, "o2": q(x2), "o3": q(b * x2)}


def slice_stats(df: pd.DataFrame, mname: str, boot: bool = True) -> dict:
    """Пункты 1–6 на срезе df (ticker, date, rv, h, k2, k3); группа = среднее по бумагам среза."""
    m = MS[mname](df)
    rv, h = df["rv"].to_numpy(float), df["h"].to_numpy(float)
    qh, qm = ql(rv, h), ql(rv, m)
    d = qh - qm
    tk = df["ticker"].to_numpy()
    per = lambda x: float(pd.Series(x).groupby(tk).mean().mean())
    q_h, q_m = per(qh), per(qm)
    best_is_h = q_h <= q_m
    qb = qh if best_is_h else qm
    under, over = (rv > h) & (rv > m), (rv < h) & (rv < m)
    inr = (rv >= np.minimum(h, m)) & (rv <= np.maximum(h, m))
    os = [o_values(rv[tk == t], h[tk == t], m[tk == t]) for t in np.unique(tk)]
    o1, o2, o3 = (float(np.mean([o[k] for o in os])) for k in ("o1", "o2", "o3"))
    q_best = min(q_h, q_m)
    div = lambda a, b: float(a / b) if b > 0 else float("nan")      # идеальный прогноз (потери 0): доли не определены, не падаем
    out = {"n_rows": len(df), "n_tickers": int(len(np.unique(tk))), "d_mean": per(d), "share_d_neg": float(np.mean(d < 0)), "n_tickers_har_better": int((pd.Series(d).groupby(tk).mean() < 0).sum()),
           "q_h": q_h, "q_m": q_m, "q_best": q_best, "best": "H" if best_is_h else "M",
           "share_both_under": float(under.mean()), "share_both_over": float(over.mean()), "loss_share_both_under": div(qb[under].sum(), qb.sum()), "loss_share_both_over": div(qb[over].sum(), qb.sum()),
           "share_in_range": float(inr.mean()), "share_outside": float(1 - inr.mean()), "loss_share_outside": div(qb[~inr].sum(), qb.sum()),
           "o1": o1, "o2": o2, "o3": o3, "gain_o1": q_best - o1, "gain_o2": q_best - o2, "gain_o3": q_best - o3, "gain_o1_pct": 100 * div(q_best - o1, q_best), "gain_o2_pct": 100 * div(q_best - o2, q_best),
           "gain_o3_pct": 100 * div(q_best - o3, q_best)}
    nan = {k: np.nan for k in ("w_har", "w_har_lo90", "w_har_hi90", "w_har01")} | {"ml_encompasses_har": np.nan, "har_encompasses_ml": np.nan}
    return out | (har_weight(rv, h, m, df["date"].to_numpy()) if boot else nan)


def slices(P: pd.DataFrame) -> list:
    """(метка, маска): годы цели 2017–2026 и окна 2017_2021, 2022_2025, all."""
    y = P["t_plus"].dt.year
    return [("Y%d" % k, (y == k).to_numpy()) for k in range(2017, 2027)] + [(w, ((y >= lo) & (y <= hi)).to_numpy()) for w, (lo, hi) in IO.WINDOWS.items() if w != "2026"]


def years_table(P: pd.DataFrame | None = None) -> pd.DataFrame:
    """DX_YEARS: срез × группа × M. P — панель 17 бумаг (общие даты H ∩ K2 ∩ K3)."""
    P = panel_rows(IO.T17) if P is None else P
    rows = []
    for lab, msk in slices(P):
        for g, sel in (("M16", P["ticker"].isin(IO.M16).to_numpy()), ("SBER", (P["ticker"] == "SBER").to_numpy())):
            df = P[msk & sel]
            if len(df) < 30:
                continue
            for mn in MS:
                rows.append({"slice": lab, "group": g, "m": mn, **slice_stats(df, mn)})
    return pd.DataFrame(rows)
