"""V5, лист 1b, G17: значимые скачки BNS (уровень 0,1 %) и концентрация потерь QLIKE (план, п. 6.5). День скачка — день цели t⁺ (в нём измерены RV и z). Диагностика задним числом.
Доли суммарного QLIKE — на общих датах M16 (как `v5_0_dx_jump`), базы H = K1, M, K8. Правило (объявлено), одновременно: (1) доля дней со значимым скачком ≤ 10 %;
(2) доля суммарного QLIKE K8 и M в эти дни ≥ 2 × доли дней; (3) не меньше 30 событий на бумагу (M16) за 2017–2021. Иначе направление 5 закрывается. 2022–2025 показывается, в решение не входит."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P, hp_stats_sel as S
from models_v2.s12 import v5_0_crit as C, v5_dx1_bns as B

BASES = {"H": "k1", "M": "M", "K8": "K8"}
WINDOWS = {"2017_2021": (2017, 2021), "2022_2025": (2022, 2025)}
MAX_SHARE, MIN_RATIO, MIN_EVENTS = 0.10, 2.0, 30


def sig_days(days: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Канонические дни (rv канона не NaN) с z и флагом sig = z > Z_CRIT."""
    ok = b.dropna(subset=["rv"])[["ticker", "date"]]
    d = ok.merge(days[["ticker", "date", "n", "z"]], on=["ticker", "date"], how="left")
    d["sig"] = d["z"] > B.Z_CRIT
    return d


def freq_rows(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d["date"].dt.year >= 2017].assign(year=lambda t: t["date"].dt.year)
    g = x.groupby(["ticker", "year"]).agg(n_days=("sig", "size"), n_sig=("sig", "sum"), n_z_nan=("z", lambda s: int(s.isna().sum()))).reset_index()
    g["freq_sig"] = g["n_sig"] / g["n_days"]
    g.insert(0, "section", "freq")
    return g


def events_rows(d: pd.DataFrame) -> pd.DataFrame:
    """События на бумагу (M16) × окно по дате цели; доля значимых дней по бумаге."""
    rows = []
    for win, (lo, hi) in WINDOWS.items():
        x = d[d["date"].dt.year.between(lo, hi) & d["ticker"].isin(P.M16)]
        g = x.groupby("ticker")["sig"].agg(["sum", "size"])
        rows += [dict(section="events", window=win, ticker=t, n_sig=int(r["sum"]), n_days=int(r["size"]), freq_sig=float(r["sum"] / r["size"])) for t, r in g.iterrows()]
    return pd.DataFrame(rows)


def conc_rows(k: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    c = S.common(k, ["k1", "k2", "k3", "d2", "M", "K8"])
    c = c[c["group"] == "M16"].merge(d[["ticker", "date", "sig"]].rename(columns={"date": "t_plus"}), on=["ticker", "t_plus"], how="left")
    c = c[c["sig"].notna()]
    rows = []
    for win, (lo, hi) in WINDOWS.items():
        w = c[c["t_plus"].dt.year.between(lo, hi)]
        m = w["sig"].to_numpy(bool)
        for base, col in BASES.items():
            q = pd.Series(C.ql(w["rv_tplus"], w[col]), index=w.index)
            under = (w["rv_tplus"] > w[col]).to_numpy()
            rows.append(dict(section="conc", base=base, window=win, share_rows=float(m.mean()), share_qlike=float(q[m].sum() / q.sum()), share_under_in=float(under[m].mean()),
                             share_under_all=float(under.mean()), qlike_mean_in=float(q[m].mean()), qlike_mean_out=float(q[~m].mean()), n_events=int(m.sum()), n_rows=len(w)))
    return pd.DataFrame(rows)


def decision(conc: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Правило п. 6.5 по окну 2017_2021: доля дней (общие даты M16), доли QLIKE K8 и M, минимум событий на бумагу M16."""
    g = conc[conc["window"] == "2017_2021"].set_index("base")
    share = float(g.loc["K8", "share_rows"])
    e = ev[ev["window"] == "2017_2021"]
    mn = int(e["n_sig"].min())
    c1, c2, c3 = share <= MAX_SHARE, bool(min(g.loc["K8", "share_qlike"], g.loc["M", "share_qlike"]) >= MIN_RATIO * share), mn >= MIN_EVENTS
    return dict(verdict="считать" if (c1 and c2 and c3) else "закрыть", share=share, ql_k8=float(g.loc["K8", "share_qlike"]), ql_m=float(g.loc["M", "share_qlike"]), ql_h=float(g.loc["H", "share_qlike"]),
                under_k8=float(g.loc["K8", "share_under_in"]), under_m=float(g.loc["M", "share_under_in"]), min_events=mn, median_events=float(e["n_sig"].median()), c1=c1, c2=c2, c3=c3)


def jump_table(k: pd.DataFrame, days: pd.DataFrame, b: pd.DataFrame) -> tuple:
    d = sig_days(days, b)
    parts = [freq_rows(d), events_rows(d), conc_rows(k, d)]
    return pd.concat(parts, ignore_index=True), decision(parts[2], parts[1])
