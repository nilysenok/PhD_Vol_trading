"""V4-B, G13: LOTO и перенос. LOTO: оператор переоценивается на 16 бумагах, исключённая получает панельные параметры (δ = 0); сравнение с пулом (полный прогон, 17 бумаг) и K1 на M16.
Перенос: панельные параметры месяца полного прогона применяются к GCHE (единственная E11 с K2/K3) без переоценки. LOTO — не новая конфигурация (в реестр не пишется)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_ops as OP, v4_b_run as R, v4_b_stats as ST, v4_io as IO


def loto_forecasts(cfg: str, P: pd.DataFrame, tickers=None) -> tuple:
    """-> (прогнозы исключённых бумаг: ticker, date, F; проверка δ = 0 по бумагам {ticker: bool})."""
    parts, zero = [], {}
    for t in tickers or IO.M16:
        out, w = R.forecast(cfg, P, exclude=t)
        parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
        zero[t] = delta_zero(cfg, w, t)
    return pd.concat(parts, ignore_index=True), zero


def delta_zero(cfg: str, w: pd.DataFrame, t: str) -> bool:
    """У исключённой бумаги параметры в каждом месяце равны панельным."""
    cols = ["wa", "wb", "wc"] if cfg == "B6" else ["w"] + (["bshift"] if cfg == "B5" else [])
    a = w[w["ticker"] == t].sort_values("month")[cols].to_numpy()
    b = w[w["ticker"] == "PANEL"].sort_values("month")[cols].to_numpy()
    return bool(np.array_equal(a, b))


def loto_table(cfg: str, lo: pd.DataFrame, full: pd.DataFrame, zero: dict) -> pd.DataFrame:
    """Бумага × окно + строка M16 (среднее по бумагам): QLIKE LOTO, пула, K1; Δ к пулу и к K1."""
    m = lo.merge(full[["ticker", "date", "F"]].rename(columns={"F": "pool"}), on=["ticker", "date"])
    rows = []
    for w in ST.WINS:
        for t, g in m.groupby("ticker"):
            g = g[ST.win_mask(g, w)]
            q = {k: ST.metr(g["rv"], g[c])["qlike"] for k, c in (("loto", "F"), ("pool", "pool"), ("k1", "h"))}
            rows.append({"config": cfg, "held_out": t, "window": w, "n": len(g), **{"q_" + k: v for k, v in q.items()}, "delta_zero_check": zero[t]})
    df = pd.DataFrame(rows)
    agg = df.groupby(["config", "window"], as_index=False)[["n", "q_loto", "q_pool", "q_k1"]].agg({"n": "sum", "q_loto": "mean", "q_pool": "mean", "q_k1": "mean"})
    agg["held_out"], agg["delta_zero_check"] = "M16", bool(all(zero.values()))
    df = pd.concat([df, agg], ignore_index=True)
    df["d_loto_pool"], df["d_loto_k1"] = df["q_loto"] - df["q_pool"], df["q_loto"] - df["q_k1"]
    df["d_loto_k1_pct_of_k1"] = 100.0 * df["d_loto_k1"] / df["q_k1"]
    return df


def transfer_gche(cfg: str, wt: pd.DataFrame) -> pd.DataFrame:
    """Панельные параметры месяца (строки PANEL полного прогона) -> GCHE. Строки: окно × {перенос, K1, K2, K3}."""
    Pg = IO.aligned("GCHE")
    mon = Pg["date"].dt.to_period("M").astype(str)
    pw = wt[wt["ticker"] == "PANEL"].set_index("month")
    F = np.empty(len(Pg))
    for m in mon.unique():
        s, q = (mon == m).to_numpy(), pw.loc[m]
        if cfg == "B6":
            th = np.tile(q[["wa", "wb", "wc"]].to_numpy(float), (int(s.sum()), 1))
            F[s] = OP.forward_b6(th, np.log(Pg["h"][s]), np.log(Pg["k2"][s]), np.log(Pg["k3"][s]))
        else:
            F[s] = OP.forward(cfg, Pg["h"].to_numpy()[s], OP.mean_m(Pg["k2"][s], Pg["k3"][s]), q["w"], float(np.nan_to_num(q["p"])), float(np.nan_to_num(q["bshift"])))[0]
    Pg = Pg.assign(F=F)
    rows = []
    for w in ST.WINS:
        g = Pg[ST.win_mask(Pg, w)]
        for name, col in (("transfer", "F"), ("k1", "h"), ("k2", "k2"), ("k3", "k3")):
            rows.append({"config": cfg, "held_out": "GCHE", "window": w, "series": name, **ST.metr(g["rv"], g[col])})
    return pd.DataFrame(rows)
