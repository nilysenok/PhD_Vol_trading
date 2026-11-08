"""V4-B: walk-forward прогноз оператора. Параметры месяца m оцениваются один раз, в первый торговый день d0 месяца, по парам, закрытым ДО d0 (t⁺ < d0), всех бумаг панели; расширяющееся окно с 2017-01.
Прогноз дня d в месяце m = оператор(H_d, M_d; параметры m). RV сюда приходит только из закрытых пар. Панель P: ticker, date, t_plus, rv, h, k2, k3 (общие даты H ∩ K2 ∩ K3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_b_ops as OP

PCOLS = ["w", "p", "bshift", "wa", "wb", "wc"]


def closed_mask(t_plus, d0):
    """Закрытые пары на начало месяца: цель t⁺ строго раньше первого дня месяца (тест подменяет на «+1»)."""
    return np.asarray(t_plus) < np.datetime64(d0)


def _month_params(cfg, P, tix, d0, nT, exclude, prev) -> tuple:
    """-> (params-массивы по бумагам (nT), n_pairs (nT,), записи, prev). Бумага без закрытых пар и исключённая получают панельные значения."""
    m = closed_mask(P["t_plus"].to_numpy(), d0)
    if exclude is not None:
        m &= tix != exclude
    npairs = np.bincount(tix[m], minlength=nT)
    pr = FT.prior_params(cfg)
    par = {"w": np.full(nT, pr["w_p"]), "p": pr["p"], "b": np.full(nT, pr["b_p"]), "th": np.tile(pr["theta_p"], (nT, 1)), "ok": True, "panel": dict(pr)}
    if m.sum() < FT.MIN_PAIRS:
        return par, npairs, prev
    T, present = FT.make_train(P["rv"].to_numpy()[m], P["h"].to_numpy()[m], P["k2"].to_numpy()[m], P["k3"].to_numpy()[m], tix[m])
    r = FT.fit(cfg, T, prev)
    par["ok"] = r["ok"]
    if cfg == "B6":
        par["th"][:] = r["theta_p"]
        par["th"][present] = r["theta"]
        par["panel"]["theta_p"] = r["theta_p"]
    else:
        par["w"][present] = r["w"]
        par["p"], par["b"][present] = r["p"], r["b"]
        par["panel"].update(w_p=r["w_p"], p=r["p"], b_p=r["b_p"])
        par["b"][np.setdiff1d(np.arange(nT), present)] = r["b_p"]
        par["w"][np.setdiff1d(np.arange(nT), present)] = r["w_p"]
    return par, npairs, r["_x"]


def _forecast_rows(cfg, X, par, tix) -> np.ndarray:
    if cfg == "B6":
        return OP.forward_b6(par["th"][tix], np.log(X["h"]), np.log(X["k2"]), np.log(X["k3"]))
    return OP.forward(cfg, X["h"], OP.mean_m(X["k2"], X["k3"]), par["w"][tix], par["p"], par["b"][tix])[0]


def _weight_rows(cfg, month, d0, tickers, par, npairs) -> list:
    pn, rows = par["panel"], []
    base = {"month": month, "d0": str(d0)[:10], "config": cfg, "converged": par["ok"]}
    th = pn["theta_p"]
    rows.append({**base, "ticker": "PANEL", "w": pn["w_p"] if cfg != "B6" else np.nan, "p": pn["p"], "bshift": pn["b_p"], "wa": th[0], "wb": th[1], "wc": th[2], "n_pairs": int(npairs.sum())})
    for i, t in enumerate(tickers):
        th = par["th"][i]
        rows.append({**base, "ticker": t, "w": par["w"][i] if cfg != "B6" else np.nan, "p": par["p"], "bshift": par["b"][i], "wa": th[0], "wb": th[1], "wc": th[2], "n_pairs": int(npairs[i])})
    return rows


def forecast(cfg: str, P: pd.DataFrame, exclude: str | None = None) -> tuple:
    """-> (прогнозы: P + F + параметры строки, веса по месяцам × бумагам). exclude — бумага, пары которой не входят в обучение (LOTO): её прогноз по панельным параметрам (δ = 0)."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    tickers = sorted(P["ticker"].unique())
    tix = P["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    ex = None if exclude is None else tickers.index(exclude)
    mon = P["date"].dt.to_period("M")
    F = np.full(len(P), np.nan)
    cols = {c: np.full(len(P), np.nan) for c in PCOLS}
    wrows, prev = [], None
    for m in mon.unique():
        sel = (mon == m).to_numpy()
        d0 = P["date"][sel].min()
        par, npairs, prev = _month_params(cfg, P, tix, d0, len(tickers), ex, prev)
        X = {"h": P["h"].to_numpy()[sel], "k2": P["k2"].to_numpy()[sel], "k3": P["k3"].to_numpy()[sel]}
        F[sel] = _forecast_rows(cfg, X, par, tix[sel])
        _fill(cols, sel, cfg, par, tix[sel])
        wrows += _weight_rows(cfg, str(m), d0, tickers, par, npairs)
    out = P.copy()
    out["F"] = F
    for c, v in cols.items():
        out[c] = v
    return _mask(cfg, out), _mask(cfg, pd.DataFrame(wrows))


def _mask(cfg, df):
    """Колонки параметров, которых у оператора нет, — NaN (p только у B4, сдвиг только у B5, a/b/c только у B6)."""
    if cfg != "B4":
        df["p"] = np.nan
    if cfg != "B5":
        df["bshift"] = np.nan
    if cfg != "B6":
        df[["wa", "wb", "wc"]] = np.nan
    return df


def _fill(cols, sel, cfg, par, ti):
    if cfg == "B6":
        cols["wa"][sel], cols["wb"][sel], cols["wc"][sel] = par["th"][ti, 0], par["th"][ti, 1], par["th"][ti, 2]
    else:
        cols["w"][sel], cols["p"][sel], cols["bshift"][sel] = par["w"][ti], par["p"], par["b"][ti]


def pit_violation(cfg: str, P: pd.DataFrame, month: str) -> bool:
    """Обрезка: подмена RV пар, закрытых не раньше d0 месяца `month`, не должна менять прогноз ни одного дня до конца этого месяца. True — нарушение (утечка из будущего)."""
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    mon = P["date"].dt.to_period("M")
    d0 = P["date"][mon == pd.Period(month)].min()
    P = P[mon <= pd.Period(month)].reset_index(drop=True)
    Q = P.copy()
    Q.loc[Q["t_plus"] >= d0, "rv"] *= 7.3
    return not np.array_equal(forecast(cfg, P)[0]["F"].to_numpy(), forecast(cfg, Q)[0]["F"].to_numpy())
