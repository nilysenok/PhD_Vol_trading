"""P1a: выходные таблицы (k23.parquet, P1_K23*.csv) и проверка записанного (G7). QLIKE — `models_v2/metrics.qlike` (основная, R12); сравнения — на общих датах (R10):
пересечение дат, где K1, K2, K3 бумаги все определены."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s10 import hp_paths as P

K23 = P.OUT / "k23.parquet"
HEAD = "# DEVELOPMENT; Ж-215/Ж-217 P1a; %s; HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок (K2/K3 замороженные, HP И2 каждой бумаги — по правилу выбора, без новой оптимизации)\n"


def write_csv(df: pd.DataFrame, name: str, note: str) -> None:
    P.REP.mkdir(parents=True, exist_ok=True)
    with open(P.REP / name, "w", encoding="utf-8") as f:
        f.write(HEAD % note)
        df.to_csv(f, index=False, lineterminator="\n")


def qlike_table(df: pd.DataFrame) -> pd.DataFrame:
    """Бумага × период цели (год, all, all_no_gap) на общих датах K1, K2, K3."""
    rows = []
    for t, g in df.groupby("ticker"):
        g = g[np.isfinite(g[["sigma2_k1", "sigma2_k2", "sigma2_k3"]]).all(axis=1)]
        masks = [("all", np.ones(len(g), bool)), ("all_no_gap", ~g["gap"].to_numpy())] + [(str(y), (g["t_plus"].dt.year == y).to_numpy()) for y in sorted(g["t_plus"].dt.year.unique())]
        for name, m in masks:
            if m.any():
                h = g[m]
                rows.append({"ticker": t, "group": P.group(t), "year": name, "n": int(m.sum()), **{"qlike_k%d" % k: float(metrics.qlike(h["rv_tplus"], h["sigma2_k%d" % k]).mean()) for k in (1, 2, 3)}})
    return pd.DataFrame(rows)


def summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Группы M16 / SBER / E11 / ALL, периоды all и all_no_gap: среднее и медиана по бумагам, доля бумаг, где K2 < K1 и K3 < K1."""
    rows = []
    for per in ("all", "all_no_gap"):
        t = tab[tab["year"] == per]
        for name, h in [(gr, t[t["group"] == gr]) for gr in ("M16", "SBER", "E11")] + [("ALL", t)]:
            if len(h):
                rows.append({"group": name, "period": per, "n_tickers": len(h), **{"mean_k%d" % k: float(h["qlike_k%d" % k].mean()) for k in (1, 2, 3)},
                             **{"median_k%d" % k: float(h["qlike_k%d" % k].median()) for k in (1, 2, 3)},
                             "share_k2_lt_k1": float((h["qlike_k2"] < h["qlike_k1"]).mean()), "share_k3_lt_k1": float((h["qlike_k3"] < h["qlike_k1"]).mean())})
    return pd.DataFrame(rows)


def sber_diff(sber: pd.DataFrame, comp: pd.DataFrame) -> dict:
    """SBER: K2 против components.XJ, K3 против components.LF — побитно на общих датах (ключ d + target_date); n — число общих строк, n_comp — всего в components."""
    m = sber.merge(comp[["d", "target_date", "rv", "XJ", "LF"]], left_on=["date", "t_plus"], right_on=["d", "target_date"], suffixes=("", "_c"))
    ok = m[np.isfinite(m["sigma2_k2"])]
    return {"n": int(len(ok)), "n_comp": int(len(comp)), "max_abs_diff_xj": float(np.abs(ok["sigma2_k2"] - ok["XJ"]).max()) if len(ok) else float("inf"),
            "max_abs_diff_lf": float(np.abs(ok["sigma2_k3"] - ok["LF"]).max()) if len(ok) else float("inf"), "max_abs_diff_rv": float(np.abs(ok["rv_tplus"] - ok["rv"]).max()) if len(ok) else float("inf")}


def verify(df: pd.DataFrame, panel: pd.DataFrame, meta: pd.DataFrame, comp: pd.DataFrame) -> list:
    """Список нарушений (пусто — G7 по записанному выходу закрыт)."""
    bad = []
    if df["ticker"].nunique() != 28:
        bad.append("бумаг %d ≠ 28" % df["ticker"].nunique())
    if len(df) != len(panel) or df.duplicated(["ticker", "date"]).any():
        bad.append("строк %d ≠ панель %d или дубли (ticker, date)" % (len(df), len(panel)))
    mm = df.merge(panel[["ticker", "date", "t_plus", "rv_tplus"]], on=["ticker", "date"], suffixes=("", "_p"))
    if len(mm) != len(df) or not (mm["t_plus"] == mm["t_plus_p"]).all() or not np.array_equal(mm["rv_tplus"].to_numpy(), mm["rv_tplus_p"].to_numpy()):
        bad.append("сетка (t, t⁺) или rv_tplus не равны panel_base побитно")
    f = df[np.isfinite(df["sigma2_k2"]) | np.isfinite(df["sigma2_k3"])]
    if not (np.isfinite(f["sigma2_k2"]) & np.isfinite(f["sigma2_k3"]) & (f["sigma2_k2"] > 0) & (f["sigma2_k3"] > 0)).all():
        bad.append("σ̂² K2/K3 не конечны или ≤ 0 на строках прогноза (или определён только один из двух)")
    y0 = meta.set_index("ticker")["y0_used"]
    early = f[f["hp_year"] < f["ticker"].map(y0)]
    if len(early):
        bad.append("прогнозы раньше первого года У-3: %d строк" % len(early))
    d = sber_diff(df[df["ticker"] == "SBER"], comp)
    if d["n"] != d["n_comp"] or d["max_abs_diff_xj"] != 0.0 or d["max_abs_diff_lf"] != 0.0 or d["max_abs_diff_rv"] != 0.0:
        bad.append("SBER не побитно: %s" % d)
    return bad
