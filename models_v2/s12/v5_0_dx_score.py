"""V5-0, G5: предварительная диагностика score s = RV/F − 1 для баз H (K1), M = sqrt(K2·K3), K8 на M16 (общие даты 2017–2021 | 2022–2025 по году цели). Диагностика задним числом: результат не выбирает гиперпараметры.
Известно на конец дня d только то, что закрыто: s пары с t⁺ = d (прогноз сделан в предыдущий торговый день). Регрессия: log(RV/F)_{t⁺} ~ const + s_свой(d) + mean s остальных бумаг(d); t — Ньюи–Уэст лаг 5 по датам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_dx_ols as O

BASES = {"H": "k1", "M": "M", "K8": "K8"}
WINDOWS = {"2017_2021": (2017, 2021), "2022_2025": (2022, 2025)}
QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
KINDS = ("s", "log")           # s = RV/F − 1 (как объявлено); log = ln(RV/F) (устойчивая к хвосту версия, рядом)


def known_scores(k: pd.DataFrame, col: str, lead: int = 0) -> pd.DataFrame:
    """Строки (бумага, d): y = ln(RV/F) цели t⁺; s_own, s_oth — score пар, закрытых к d (t⁺ = d); lead — только для мутации (score пар t⁺ = d + lead)."""
    w = k.copy()
    w["ls"] = np.log(w["rv_tplus"] / w[col])
    w["s"] = w["rv_tplus"] / w[col] - 1.0
    key = w[["ticker", "t_plus", "s", "ls"]].rename(columns={"t_plus": "date", "s": "s_own", "ls": "ls_own"})
    w = w.merge(key, on=["ticker", "date"], how="left").sort_values(["ticker", "date"]).reset_index(drop=True)
    if lead:       # мутация: score пары, закрывающейся через `lead` торговых дней (для lead = 1 — пары самой строки)
        w[["s_own", "ls_own"]] = w.groupby("ticker")[["s_own", "ls_own"]].shift(-lead)
        key = w[["ticker", "date", "s_own", "ls_own"]]
    for a in ("s", "ls"):
        tot = key.groupby("date")[a + "_own"].agg(["sum", "count"])
        sm, cn = w["date"].map(tot["sum"]), w["date"].map(tot["count"])
        w[a + "_oth"] = (sm - w[a + "_own"]) / (cn - 1)
    return w


def dist_rows(w: pd.DataFrame, base: str, win: str) -> list:
    rows = []
    for kind, col in (("s", "s"), ("log", "ls")):
        x = w[col].dropna()
        rows += [dict(base=base, window=win, kind=kind, metric="q%02d" % round(100 * q), value=float(x.quantile(q))) for q in QS]
        rows += [dict(base=base, window=win, kind=kind, metric="mean", value=float(x.mean())), dict(base=base, window=win, kind=kind, metric="share_s_gt_5", value=float((w["s"] > 5).mean()))]
    return rows


def acf_rows(w: pd.DataFrame, base: str, win: str) -> list:
    rows = []
    for kind, col in (("s", "s"), ("log", "ls")):
        for lag in range(1, 6):
            v = [g.sort_values("t_plus")[col].autocorr(lag) for _, g in w.groupby("ticker")]
            rows.append(dict(base=base, window=win, kind=kind, metric="acf_lag%d_mean_over_tickers" % lag, value=float(np.nanmean(v))))
        piv = w.pivot_table(index="t_plus", columns="ticker", values=col).corr().to_numpy()
        rows.append(dict(base=base, window=win, kind=kind, metric="mean_cross_corr", value=float((piv.sum() - len(piv)) / (len(piv) * (len(piv) - 1)))))
    return rows


def reg_rows(w: pd.DataFrame, base: str, win: str) -> list:
    rows = []
    for kind, own, oth in (("s", "s_own", "s_oth"), ("log", "ls_own", "ls_oth")):
        d = w.dropna(subset=["ls", own, oth])
        X = np.column_stack([np.ones(len(d)), d[own], d[oth]])
        r = O.nw_ols(d["ls"].to_numpy(), X, d["date"].to_numpy())
        for i, nm in ((1, "own"), (2, "oth")):
            rows += [dict(base=base, window=win, kind=kind, metric="beta_" + nm, value=float(r["beta"][i])), dict(base=base, window=win, kind=kind, metric="t_nw_" + nm, value=float(r["t"][i]))]
        rows += [dict(base=base, window=win, kind=kind, metric="r2", value=r["r2"]), dict(base=base, window=win, kind=kind, metric="n", value=float(r["n"]))]
    return rows


def score_table(k: pd.DataFrame, lead: int = 0) -> pd.DataFrame:
    """PASS_SCORE: base × окно × вид × метрика."""
    c = S.common(k, ["k1", "k2", "k3", "d2", "M", "K8"])
    c = c[c["group"] == "M16"]
    rows = []
    for base, col in BASES.items():
        for win, (lo, hi) in WINDOWS.items():
            cc = c[c["t_plus"].dt.year.between(lo, hi)]
            w = known_scores(cc, col, lead)
            rows += dist_rows(w, base, win) + acf_rows(w, base, win) + reg_rows(w, base, win)
    return pd.DataFrame(rows)
