"""V4-CLD: walk-forward прогноз CLD1–CLD4. F = H·exp(w·ℓ), ℓ = ln(M/H), M = √(K2·K3), H = K1; w = g(D̂_{d+1}), D̂ = E[D | z_d], D = QLIKE(H) − QLIKE(M) на цели пары (плюс — ML лучше).
Регрессия — раз в месяц, в первый торговый день d0, по парам, закрытым ДО d0 (t⁺ < d0; `v4_b_run.closed_mask`), всех бумаг панели, расширяющееся окно с 2017-01; признаки строки d известны на конец d (Ж-225 п. 5: меток дня цели нет).
До первых 60 закрытых пар панели (с конечными признаками) w = 0,5. Строка обучения с нефинитным признаком исключается; в прогнозе нефинитный признак = 0 после стандартизации (среднее окна).
Признаки — весь объявленный список без отбора: FEATS (+ время после излома `brk` из CUSUM по медиане 17 — в LOTO по 16, `v4_cld_brk`). Исключённая бумага (LOTO) не входит в обучение; её z — по 16. `leak`, `lead_brk`, `force_dhat` — только тесты."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_c_run as CR, v4_cld_brk as BK, v4_cld_fit as CF, v4_c_fit as CFIT, v4_z_feat as ZF

FEATS = ["ell", "abs_ell", "lerr_h", "lerr_m", "lrv", "med_lrv", "dlrv", "herf", "lasth", "sur_mkt", "brk"]
CFGS = CF.CFGS


def _ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def prepare(P: pd.DataFrame, Zl: pd.DataFrame, lead_brk: int = 0, leak: bool = False) -> tuple:
    """-> (P по (date, ticker), X (N, k) признаки, имена, tix, тикеры, D — реализованный дифференциал на цели пары). leak — ТОЛЬКО мутация: D цели в признаках."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    base = [f for f in FEATS if f != "brk"]
    Z = P[["ticker", "date"]].merge(Zl[["ticker", "date"] + base], on=["ticker", "date"], how="left")
    Z["brk"] = P["date"].map(BK.brk_by_date(Zl, lead_brk)).to_numpy(float)
    M = np.sqrt(P["k2"].to_numpy(float) * P["k3"].to_numpy(float))
    D = _ql(P["rv"], P["h"]) - _ql(P["rv"], M)
    names = list(FEATS)
    if leak:
        Z["d_future"], names = D, names + ["d_future"]
    tk = sorted(P["ticker"].unique())
    return P, Z[names].to_numpy(float), names, P["ticker"].map({t: i for i, t in enumerate(tk)}).to_numpy(), tk, D


def _fit_month(cfg, X, D, closed, ex_mask) -> dict | None:
    m = closed & ex_mask & np.isfinite(X).all(1) & np.isfinite(D)
    if m.sum() < FT.MIN_PAIRS:
        return None
    mu, sd = CFIT.standardize(X[m])
    r = CF.fit_model(cfg, (X[m] - mu) / sd, D[m])
    return {**r, "mu": mu, "sd": sd, "n_train": int(m.sum()), "dbar": float(D[m].mean())}


def _predict(cfg, mod, X, sel, force_dhat) -> tuple:
    n = int(sel.sum())
    if mod is None:
        return np.zeros(n), np.full(n, 0.5), np.nan
    z = np.where(np.isfinite(X[sel]), (X[sel] - mod["mu"]) / mod["sd"], 0.0)
    dh = mod["predict"](z) if force_dhat is None else np.full(n, float(force_dhat))
    return dh, CF.weight(cfg, dh, mod["s"], mod["c"]), mod["dbar"]


def forecast(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, exclude: str | None = None, lead_brk: int = 0, leak: bool = False, force_dhat: float | None = None) -> tuple:
    """-> (out: ticker, date, t_plus, rv, h, F, w, dhat, dbar, d_act; веса config × month × ticker (PANEL + бумаги); коэффициенты config × month × feature)."""
    P, X, names, tix, tk, D = prepare(P, Zl, lead_brk, leak)
    ex = None if exclude is None else tk.index(exclude)
    mon = P["date"].dt.to_period("M")
    dhat, w, dbar = (np.full(len(P), np.nan) for _ in range(3))
    wrows, crows = [], []
    for p in mon.unique():
        sel = (mon == p).to_numpy()
        d0 = P["date"][sel].min()
        closed = BR.closed_mask(P["t_plus"].to_numpy(), d0)
        mod = _fit_month(cfg, X, D, closed, tix != ex)
        dhat[sel], w[sel], dbar[sel] = _predict(cfg, mod, X, sel, force_dhat)
        wrows += _wrows(cfg, str(p), d0, tk, tix, sel, w, ex, int((closed & (tix != ex)).sum()))
        crows += _crows(cfg, str(p), names, mod)
    lH = np.log(P["h"].to_numpy(float))
    ell = np.log(np.sqrt(P["k2"].to_numpy(float) * P["k3"].to_numpy(float))) - lH
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["w"], out["dhat"], out["dbar"], out["d_act"] = np.exp(lH + w * ell), w, dhat, dbar, D
    return out, pd.DataFrame(wrows), pd.DataFrame(crows)


def _wrows(cfg, month, d0, tk, tix, sel, w, ex, npairs) -> list:
    base = {"config": cfg, "month": month, "d0": str(d0)[:10]}
    rows = [{**base, "ticker": "PANEL", "mean_w": float(np.mean(w[sel])), "n_pairs": npairs}]
    for i, t in enumerate(tk):
        s = sel & (tix == i)
        rows.append({**base, "ticker": t, "mean_w": float(np.mean(w[s])) if s.any() else np.nan, "excluded": bool(ex == i)})
    return rows


def _crows(cfg, month, names, mod) -> list:
    base = {"config": cfg, "month": month, "n_train": 0 if mod is None else mod["n_train"], "s_hat": np.nan if mod is None else mod["s"], "c90": np.nan if mod is None else mod["c"],
            "converged": True if mod is None else mod["ok"]}
    return [{**base, "feature": f, "coef": np.nan if mod is None else mod["coef"][j]} for j, f in enumerate(names)]


def pit_violation(cfg: str, P: pd.DataFrame, inputs: dict, day, scale: float = 7.3, **kw) -> bool:
    """Обрезка боевого пути (Z + brk + оценка + прогноз): подмена RV пар с t⁺ > day, дневных величин и herf после day, отбрасывание строк после day не меняют w, F ни одной строки с date <= day.
    day — первый торговый день месяца (ловит «закрыто + 1»). kw — мутации (`leak`, `lead_brk`) в forecast; `lead` — признаки дня d+lead. True — нарушение."""
    day, lead = pd.Timestamp(day), kw.pop("lead", 0)
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    alt = {t: {"daily": CR._scale_after(v["daily"], day, scale), "herf": v["herf"].where(v["herf"].index <= day, 0.5)} for t, v in inputs.items()}
    Pc = P[P["date"] <= day].reset_index(drop=True)
    fa, fb, fc = (forecast(cfg, X, ZF.build_panel_z(X, inp, lead=lead), **kw)[0] for X, inp in ((P, inputs), (B, alt), (Pc, inputs)))
    m = (fa["date"] <= day).to_numpy()
    same = lambda x, y: all(np.array_equal(x[c].to_numpy(), y[c].to_numpy()) for c in ("F", "w"))
    return not (same(fa[m], fb[m]) and same(fa[m], fc))
