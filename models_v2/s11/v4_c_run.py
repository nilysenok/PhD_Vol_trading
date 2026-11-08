"""V4-C: walk-forward прогноз условной смеси. Параметры месяца m оцениваются один раз, в первый торговый день d0 месяца, по парам, закрытым ДО d0 (t⁺ < d0; `v4_b_run.closed_mask`), всех бумаг панели,
расширяющееся окно с 2017-01. Признаки строки d известны на конец d. До первых 60 закрытых пар — приор (w = 0,5). Исключённая бумага (LOTO) не входит в оценку и получает δ = 0.
Строка обучения с нефинитным признаком из оценки исключается (не заполняется); в прогнозе нефинитный признак после стандартизации = 0 (среднее окна) — число таких строк пишется в вес-таблицу."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_run as BR, v4_c_fit as CF, v4_z_feat as ZF


def _prep(P: pd.DataFrame, Zl: pd.DataFrame, feats: list) -> tuple:
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    X = P[["ticker", "date"]].merge(Zl[["ticker", "date"] + feats], on=["ticker", "date"], how="left")[feats].to_numpy(float)
    return P, X


def _x0(prev, present, k):
    if prev is None:
        return None
    d = np.array([prev["delta"].get(t, 0.0) for t in present])
    return np.concatenate([[prev["alpha"]], d, prev["beta"]])


def _fit_month(P, X, tix, tickers, d0, ex, prev) -> tuple:
    k = X.shape[1]
    m = BR.closed_mask(P["t_plus"].to_numpy(), d0) & (tix != ex) & np.isfinite(X).all(1)
    npairs = np.bincount(tix[BR.closed_mask(P["t_plus"].to_numpy(), d0) & (tix != ex)], minlength=len(tickers))
    base = {"alpha": 0.0, "delta": np.zeros(len(tickers)), "beta": np.zeros(k), "mu": np.zeros(k), "sd": np.ones(k), "ok": True, "grad": 0.0, "nit": 0, "n_train": int(m.sum())}
    if m.sum() < FT.MIN_PAIRS:
        return base, npairs, prev
    mu, sd = CF.standardize(X[m])
    present, tid = np.unique(tix[m], return_inverse=True)
    lH = np.log(P["h"].to_numpy(float)[m])
    ell = np.log(np.sqrt(P["k2"].to_numpy(float)[m] * P["k3"].to_numpy(float)[m])) - lH
    names = [tickers[i] for i in present]
    r = CF.fit(P["rv"].to_numpy(float)[m], lH, ell, (X[m] - mu) / sd, tid, len(present), _x0(prev, names, k))
    d = np.zeros(len(tickers))
    d[present] = r["delta"]
    base.update(alpha=r["alpha"], delta=d, beta=r["beta"], mu=mu, sd=sd, ok=r["ok"], grad=r["grad"], nit=r["nit"])
    return base, npairs, {"alpha": r["alpha"], "beta": r["beta"], "delta": dict(zip(names, r["delta"]))}


def _predict(par, P, X, tix, sel) -> tuple:
    z = np.where(np.isfinite(X[sel]), (X[sel] - par["mu"]) / par["sd"], 0.0)
    eta = par["alpha"] + par["delta"][tix[sel]] + (CF.mv(z, par["beta"]) if z.shape[1] else 0.0)
    lH = np.log(P["h"].to_numpy(float)[sel])
    ell = np.log(np.sqrt(P["k2"].to_numpy(float)[sel] * P["k3"].to_numpy(float)[sel])) - lH
    F, w = CF.forward(lH, ell, eta)
    return F, w, int((~np.isfinite(X[sel])).any(1).sum())


def _wrows(cfg, feats, month, d0, tickers, par, npairs, P, sel, w, tix, ex, n_nan) -> list:
    base = {"config": cfg, "month": month, "d0": str(d0)[:10], "converged": par["ok"]}
    row = {**base, "ticker": "PANEL", "alpha": par["alpha"], "delta": np.nan, "mean_w": float(np.mean(w)), "n_pairs": int(npairs.sum()), "n_train": par["n_train"], "grad": par["grad"], "nit": par["nit"],
           "n_rows_nan_feature": n_nan, **{"b_" + f: np.nan for f in ZF.ALL}}
    row.update({"b_" + f: float(b) for f, b in zip(feats, par["beta"])})
    out = [row]
    for i, t in enumerate(tickers):
        s = tix[sel] == i
        out.append({**base, "ticker": t, "alpha": par["alpha"] + par["delta"][i], "delta": par["delta"][i], "mean_w": float(np.mean(w[s])) if s.any() else np.nan, "n_pairs": int(npairs[i]),
                    "excluded": bool(ex == i)})
    return out


def forecast(cfg: str, P: pd.DataFrame, Zl: pd.DataFrame, exclude: str | None = None, ablate: dict | None = None) -> tuple:
    """-> (прогнозы: ticker, date, t_plus, rv, h, F, w; веса по месяцам × бумагам с β панели). Zl — длинная таблица признаков (поперечные колонки уже по нужному набору бумаг).
    ablate {имя: [признаки]} — ДИАГНОСТИКА, не конфигурация: колонки F_abl_<имя> = прогноз той же оценки с обнулёнными коэффициентами этих признаков (без переоценки)."""
    feats = ZF.cfg_feats(cfg)
    ablate = ablate or {}
    P, X = _prep(P, Zl, feats)
    tickers = sorted(P["ticker"].unique())
    tix = P["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    ex = None if exclude is None else tickers.index(exclude)
    mon = P["date"].dt.to_period("M")
    F, W = np.full(len(P), np.nan), np.full(len(P), np.nan)
    FA = {a: np.full(len(P), np.nan) for a in ablate}
    wrows, prev = [], None
    for m in mon.unique():
        sel = (mon == m).to_numpy()
        d0 = P["date"][sel].min()
        par, npairs, prev = _fit_month(P, X, tix, tickers, d0, ex, prev)
        F[sel], W[sel], n_nan = _predict(par, P, X, tix, sel)
        for a, fs in ablate.items():
            b = par["beta"].copy()
            b[[feats.index(f) for f in fs]] = 0.0
            FA[a][sel] = _predict({**par, "beta": b}, P, X, tix, sel)[0]
        wrows += _wrows(cfg, feats, str(m), d0, tickers, par, npairs, P, sel, W[sel], tix, ex, n_nan)
    out = P[["ticker", "date", "t_plus", "rv", "h"]].copy()
    out["F"], out["w"] = F, W
    for a, v in FA.items():
        out["F_abl_" + a] = v
    return out, pd.DataFrame(wrows)


def pit_violation(cfg: str, P: pd.DataFrame, inputs: dict, day, scale: float = 7.3, lead: int = 0, herf_lag: int = 0) -> bool:
    """Обрезка боевого пути (признаки + оценка + прогноз) на полной панели P: подмена RV пар с t⁺ > day, дневных величин и herf после day не должна менять прогноз ни одной строки с date <= day.
    True — нарушение (утечка из будущего). day берётся первым торговым днём месяца, чтобы ловить и «закрыто + 1» в оценке параметров. lead / herf_lag — только для тестов-мутаций
    («признак дня d+1», «Intraday Shape из баров d+1»: сдвиг применяется ПОСЛЕ подмены, как в `herf_by_day(lag=1)`); боевое значение 0."""
    day = pd.Timestamp(day)
    B = P.copy()
    B.loc[B["t_plus"] > day, "rv"] *= scale
    alt = {t: {"daily": _scale_after(v["daily"], day, scale), "herf": v["herf"].where(v["herf"].index <= day, 0.5)} for t, v in inputs.items()}
    lag = (lambda inp: {t: {**v, "herf": v["herf"].shift(-herf_lag)} for t, v in inp.items()}) if herf_lag else (lambda inp: inp)
    fa = forecast(cfg, P, ZF.build_panel_z(P, lag(inputs), lead=lead))[0]
    fb = forecast(cfg, B, ZF.build_panel_z(B, lag(alt), lead=lead))[0]
    m = (fa["date"] <= day).to_numpy()
    return not np.array_equal(fa["F"].to_numpy()[m], fb["F"].to_numpy()[m])


def _scale_after(daily: pd.DataFrame, day, scale: float) -> pd.DataFrame:
    d = daily.copy()
    late = d.index > day
    d.loc[late, ["rv", "j", "rv_close"]] = d.loc[late, ["rv", "j", "rv_close"]].to_numpy() * scale
    return d
