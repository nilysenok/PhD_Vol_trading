"""V4-CS: walk-forward прогноз CS1, CS2 — C8 (два режима, признак med_lrv, месячная переоценка в первый торговый день d0 по парам t⁺ < d0 всех бумаг панели; до 60 закрытых пар — приор w = 0,5) с убывающими штрафами
(`v4_cs_fit.penalties`). Накопленные пары n — только закрытые к d0 (`count_pairs`, подменяется мутацией «κ_i от будущих пар»). Исключённая бумага (LOTO) не входит ни в оценку, ни в счёт пар.
Диагностика (`v4_dx_*`) сюда не импортируется (тест изоляции)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_c_fit as CF, v4_c_run as CR, v4_cb_models as MD, v4_cb_run as CB, v4_cs_fit as FS, v4_z_feat as ZF

CFGS = ["CS1", "CS2"]
FEATS = ["med_lrv"]                    # как у C8 (v4_cb_run.SPECS["C8"])


def count_pairs(closed, tix, nT) -> np.ndarray:
    """Закрытые пары по бумагам (closed — уже маска t⁺ < d0 без исключённой). Мутация подменяет на счёт всех строк (будущие пары)."""
    return np.bincount(tix[closed], minlength=nT)


def _fit_period(cfg, P, X, tix, tickers, d0, ex) -> tuple:
    closed = BR.closed_mask(P["t_plus"].to_numpy(), d0) & (tix != ex)
    m = closed & np.isfinite(X).all(1)
    npairs = count_pairs(closed, tix, len(tickers))
    if m.sum() < FT.MIN_PAIRS:
        return MD.prior_state("regime", len(tickers), 1), npairs
    present, tid = np.unique(tix[m], return_inverse=True)
    tr = {**CB._arr(P, X, m), "rv": P["rv"].to_numpy(float)[m]}
    mu, sd = CF.standardize(tr["X"])
    kappa_i, lam = FS.penalties(cfg, int(npairs.sum()), npairs[present])
    r = FS.fit_regime_vec(tr["rv"], tr["lH"], tr["ell"], ((tr["X"] - mu) / sd)[:, 0], tid, len(present), kappa_i, lam)
    nT = len(tickers)
    r.update(mu=mu, sd=sd, d_calm=MD._expand(r["d_calm"], present, nT), d_crisis=MD._expand(r["d_crisis"], present, nT), n_train=int(m.sum()), lam=lam, kappa_mean=float(np.mean(kappa_i)))
    return r, npairs


def _wrows(cfg, per, d0, tickers, s, npairs, w, tix, ex) -> list:
    base = {"config": cfg, "month": per, "d0": str(d0)[:10], "converged": s["ok"]}
    out = [{**base, "ticker": "PANEL", "mean_w": float(np.mean(w)), "n_pairs": int(npairs.sum()), "n_train": s.get("n_train", 0), "grad": s["grad"], "nit": s["nit"], "lam": s.get("lam", FS.K0),
            "kappa_mean": s.get("kappa_mean", FS.K0), **MD.panel("regime", s, [])}]
    for i, t in enumerate(tickers):
        q = tix == i
        out.append({**base, "ticker": t, "alpha": MD.alpha_i("regime", s, i), "dev": MD.dev("regime", s, i), "mean_w": float(np.mean(w[q])) if q.any() else np.nan, "n_pairs": int(npairs[i]),
                    "kappa_i": float(FS.sched(npairs[i])) if cfg == "CS2" else FS.K0, "excluded": bool(ex == i)})
    return out


def forecast(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, exclude: str | None = None) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, w, pi; веса по месяцам × бумагам). Zl — длинная таблица признаков (поперечные колонки уже по нужному набору бумаг)."""
    P, X = CR._prep(P, Zl, FEATS)
    tickers = sorted(P["ticker"].unique())
    tix = P["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    ex = None if exclude is None else tickers.index(exclude)
    per = P["date"].dt.to_period("M")
    F, W, PI = (np.full(len(P), np.nan) for _ in range(3))
    wrows = []
    for p in per.unique():
        sel = (per == p).to_numpy()
        d0 = P["date"][sel].min()
        s, npairs = _fit_period(cfg, P, X, tix, tickers, d0, ex)
        pr = {**CB._arr(P, X, sel), "tix": tix[sel]}
        F[sel], W[sel], extra = MD.predict_prior("regime", pr) if s.get("prior") else MD.predict("regime", s, pr)
        PI[sel] = extra["pi"]
        wrows += _wrows(cfg, str(p), d0, tickers, s, npairs, W[sel], tix[sel], ex)
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["w"], out["pi"] = F, W, PI
    return out, pd.DataFrame(wrows)


def pit_violation(cfg: str, P: pd.DataFrame, inputs: dict, day, scale: float = 7.3) -> bool:
    """Обрезка боевого пути: (а) подмена RV пар с t⁺ > day и дневных величин после day и (б) отбрасывание всех строк после day не меняют прогноз ни одной строки с date <= day. True — нарушение.
    day — первый торговый день месяца: ловит «закрыто + 1» (а) и счёт будущих пар в κ_i / λ (б)."""
    day = pd.Timestamp(day)
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    alt = {t: {"daily": CR._scale_after(v["daily"], day, scale), "herf": v["herf"].where(v["herf"].index <= day, 0.5)} for t, v in inputs.items()}
    Pc = P[P["date"] <= day].reset_index(drop=True)
    fa = forecast(cfg, P, ZF.build_panel_z(P, inputs))[0]
    fb = forecast(cfg, B, ZF.build_panel_z(B, alt))[0]
    fc = forecast(cfg, Pc, ZF.build_panel_z(Pc, inputs))[0]
    m = (fa["date"] <= day).to_numpy()
    return not (np.array_equal(fa["F"].to_numpy()[m], fb["F"].to_numpy()[m]) and np.array_equal(fa["F"].to_numpy()[m], fc["F"].to_numpy()))
