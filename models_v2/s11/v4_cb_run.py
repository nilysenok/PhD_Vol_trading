"""V4-C, лист C-b: walk-forward прогноз C5–C10. Как в C-a (`v4_c_run`): параметры периода оцениваются один раз, в первый торговый день d0 периода, по парам, закрытым ДО d0 (t⁺ < d0; `v4_b_run.closed_mask`),
всех бумаг панели (расширяющееся окно с 2017-01); до первых 60 закрытых пар — приор. Период: месяц (C5, C6, C8, C9, C10) или год (C7: «переобучение раз в год, в первый торговый день года»).
Исключённая бумага (LOTO) не входит в оценку и получает панельные параметры (отклонения 0). Строка обучения с нефинитным признаком из оценки исключается; в прогнозе нефинитный признак = среднее/медиана окна."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_cb_models as MD, v4_c_run as CR, v4_z_feat as ZF

# cfg -> (kind, признаки из Z, период). C1x — не конфигурация программы: линейный контур с набором C1, нужен только для проверки тождества с C1 (тест, --check).
SPECS = {"C1x": ("lin", ZF.cfg_feats("C1"), "M"), "C5": ("gam", ["ell"], "M"), "C6": ("mono", ["ell", "abs_ell", "ewqd"], "M"), "C7": ("boost", ZF.cfg_feats("C4"), "Y"),
         "C8": ("regime", ["med_lrv"], "M"), "C9": ("three", [], "M"), "C10": ("eta", ZF.cfg_feats("C1"), "M")}
CFGS = ["C5", "C6", "C7", "C8", "C9", "C10"]
BNAMES = {"gam": ["s%d" % i for i in range(1, 6)]}


def _arr(P, X, sel) -> dict:
    lH = np.log(P["h"].to_numpy(float)[sel])
    k2, k3 = P["k2"].to_numpy(float)[sel], P["k3"].to_numpy(float)[sel]
    return {"lH": lH, "lK2": np.log(k2), "lK3": np.log(k3), "ell": np.log(np.sqrt(k2 * k3)) - lH, "X": X[sel]}


def _fit_period(kind, P, X, tix, tickers, d0, ex, prev) -> tuple:
    closed = BR.closed_mask(P["t_plus"].to_numpy(), d0) & (tix != ex)
    m = closed & np.isfinite(X).all(1)
    npairs = np.bincount(tix[closed], minlength=len(tickers))
    if m.sum() < FT.MIN_PAIRS:
        return MD.prior_state(kind, len(tickers), X.shape[1]), npairs, prev
    present, tid = np.unique(tix[m], return_inverse=True)
    tr = {**_arr(P, X, m), "rv": P["rv"].to_numpy(float)[m], "tid": tid, "n": len(present), "present": present, "names": [tickers[i] for i in present], "nT": len(tickers)}
    s, prev = MD.fit(kind, tr, prev)
    s["n_train"] = int(m.sum())
    return s, npairs, prev


def _wrows(cfg, kind, per, d0, tickers, s, npairs, w, tix, ex, n_nan) -> list:
    base = {"config": cfg, "month": per, "d0": str(d0)[:10], "converged": s["ok"]}
    out = [{**base, "ticker": "PANEL", "mean_w": float(np.mean(w)), "n_pairs": int(npairs.sum()), "n_train": s.get("n_train", 0), "grad": s["grad"], "nit": s["nit"], "n_rows_nan_feature": n_nan,
            **MD.panel(kind, s, BNAMES.get(kind, SPECS[cfg][1]))}]
    for i, t in enumerate(tickers):
        q = tix == i
        out.append({**base, "ticker": t, "alpha": MD.alpha_i(kind, s, i), "dev": MD.dev(kind, s, i), "mean_w": float(np.mean(w[q])) if q.any() else np.nan, "n_pairs": int(npairs[i]), "excluded": bool(ex == i)})
    return out


def forecast(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, exclude: str | None = None) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, w [+ дополнительные колонки модели]; веса по периодам × бумагам). Zl — длинная таблица признаков (поперечные колонки уже по нужному набору бумаг)."""
    kind, feats, freq = SPECS[cfg]
    P, X = CR._prep(P, Zl, feats)
    tickers = sorted(P["ticker"].unique())
    tix = P["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    ex = None if exclude is None else tickers.index(exclude)
    per = P["date"].dt.to_period(freq)
    F, W, extra = np.full(len(P), np.nan), np.full(len(P), np.nan), {}
    wrows, prev = [], None
    for p in per.unique():
        sel = (per == p).to_numpy()
        d0 = P["date"][sel].min()
        s, npairs, prev = _fit_period(kind, P, X, tix, tickers, d0, ex, prev)
        pr = {**_arr(P, X, sel), "tix": tix[sel]}
        F[sel], W[sel], ex_c = MD.predict_prior(kind, pr) if s.get("prior") else MD.predict(kind, s, pr)
        for c, v in ex_c.items():
            extra.setdefault(c, np.full(len(P), np.nan))[sel] = v
        wrows += _wrows(cfg, kind, str(p), d0, tickers, s, npairs, W[sel], tix[sel], ex, int((~np.isfinite(X[sel])).any(1).sum()))
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["w"] = F, W
    for c, v in extra.items():
        out[c] = v
    return out, pd.DataFrame(wrows)


def pit_violation(cfg: str, P: pd.DataFrame, inputs: dict, day, scale: float = 7.3) -> bool:
    """Обрезка боевого пути (признаки + оценка + прогноз): подмена RV пар с t⁺ > day и дневных величин/herf после day не должна менять прогноз ни одной строки с date <= day. True — нарушение.
    day — первый торговый день периода модели (месяц/год), чтобы ловить и «закрыто + 1» в оценке параметров."""
    day = pd.Timestamp(day)
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    alt = {t: {"daily": CR._scale_after(v["daily"], day, scale), "herf": v["herf"].where(v["herf"].index <= day, 0.5)} for t, v in inputs.items()}
    fa = forecast(cfg, P, ZF.build_panel_z(P, inputs))[0]
    fb = forecast(cfg, B, ZF.build_panel_z(B, alt))[0]
    m = (fa["date"] <= day).to_numpy()
    return not np.array_equal(fa["F"].to_numpy()[m], fb["F"].to_numpy()[m])
