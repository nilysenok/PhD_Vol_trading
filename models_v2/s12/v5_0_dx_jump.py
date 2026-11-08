"""V5-0, G6: концентрация потерь в дни скачков (условие направления 5). Скачок J > 0 — по определению проекта (`daily.build_daily`: J = max(RV − BV, 0), BV = (π/2)·Σ|r_i||r_{i-1}|). Диагностика задним числом: день цели t⁺ — день, в который измерены RV и J.
Доли — от суммарного QLIKE базы (H = K1, M, K8) на M16, общие даты; «топ 1 % / 5 %» — дни с наибольшей RV в бумаге и окне."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_crit as C, v5_0_dx_score as SC

BIG = 0.5       # «крупный» скачок: J/RV > 0,5 (сверх определения; для частоты крупных событий)


def freq_rows(b: pd.DataFrame) -> pd.DataFrame:
    """Частота дней с J > 0, число событий (всех и крупных) на бумагу × год (канонические дни, 2017–2025 + 2026 неполный)."""
    d = b.dropna(subset=["rv"])
    d = d[d["date"].dt.year >= 2017].assign(year=lambda x: x["date"].dt.year, jump=lambda x: x["j"] > 0, big=lambda x: x["j"] / x["rv"] > BIG)
    g = d.groupby(["ticker", "year"]).agg(n_days=("rv", "size"), n_jump=("jump", "sum"), n_big=("big", "sum")).reset_index()
    g["freq_jump"] = g["n_jump"] / g["n_days"]
    g.insert(0, "section", "freq")
    return g


def conc_rows(k: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    c = S.common(k, ["k1", "k2", "k3", "d2", "M", "K8"])
    c = c[c["group"] == "M16"].merge(b[["ticker", "date", "j"]].rename(columns={"date": "t_plus"}), on=["ticker", "t_plus"], how="left")
    rows = []
    for win, (lo, hi) in SC.WINDOWS.items():
        w = c[c["t_plus"].dt.year.between(lo, hi)].copy()
        w["top1"] = w.groupby("ticker")["rv_tplus"].transform(lambda s: s >= s.quantile(0.99))
        w["top5"] = w.groupby("ticker")["rv_tplus"].transform(lambda s: s >= s.quantile(0.95))
        w["jump"] = w["j"] > 0
        for base, col in SC.BASES.items():
            q = pd.Series(C.ql(w["rv_tplus"], w[col]), index=w.index)
            under = w["rv_tplus"] > w[col]
            for flag in ("jump", "top1", "top5"):
                m = w[flag].to_numpy(bool)
                rows.append(dict(section="conc", base=base, window=win, subset=flag, share_rows=float(m.mean()), share_qlike=float(q[m].sum() / q.sum()), share_under=float(under[m].mean()),
                                 share_under_all=float(under.mean()), qlike_mean_in=float(q[m].mean()), qlike_mean_out=float(q[~m].mean()), n=int(m.sum())))
    return pd.DataFrame(rows)


def jump_table(k: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([freq_rows(b), conc_rows(k, b)], ignore_index=True)
