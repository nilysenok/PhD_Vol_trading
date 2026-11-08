"""V4-C, лист C-b, G20: `crit_panel` по годам цели. Те же строки и тот же способ, что внутри `hp_stats_sel.crit_panel` / `d_comp.crit_wf`: общие даты всех рядов (S.common), M16, окно выбора 2017–2021 по году цели;
ann_y — средний QLIKE бумаги за год цели; pen_Y = mean_{y<Y} ann_y + 0,25·(max − mean) для Y = 2018…2021; crit_wf бумаги = mean_Y pen_Y; crit_panel = среднее по бумагам. Здесь по бумагам усредняются ann_y и pen_Y
(линейно), поэтому crit_part = pen_Y / 4 даёт crit_panel суммой по годам (проверка тождества — в `--check`). Годы 2022–2026 — только ann_y (описательно; в выбор не входят)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.xgb.d_comp import PENALTY
from models_v2.s10 import hp_stats_sel as S

SEL_YEARS = range(2017, 2022)
DESC_YEARS = range(2022, 2027)


def _pen(ann: dict) -> dict:
    """pen_Y по ann бумаги {год: средний QLIKE} (как `crit_wf`)."""
    out = {}
    for Y in sorted(ann):
        past = [v for y, v in ann.items() if y < Y]
        if past:
            m = float(np.mean(past))
            out[Y] = m + PENALTY * (max(past) - m)
    return out


def _qlike_table(m16: pd.DataFrame, cols: list) -> pd.DataFrame:
    Q = pd.DataFrame({c: m16["rv_tplus"].to_numpy(float) / m16[c].to_numpy(float) for c in cols})
    Q = Q - np.log(Q) - 1.0
    Q["ticker"], Q["year"] = m16["ticker"].to_numpy(), m16["t_plus"].dt.year.to_numpy()
    return Q


def years_table(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    """-> строки (series, year, role, n_tickers, ann_qlike, pen, crit_part, d_ann_k1, d_ann_d2); k — широкая таблица оценки (`v4_b_data.eval_frame`)."""
    c = S.common(k, cols)
    m16 = c[c["group"] == "M16"]
    win = m16[(m16["t_plus"].dt.year >= SEL_YEARS[0]) & (m16["t_plus"].dt.year <= SEL_YEARS[-1])]
    Qw, Qa = _qlike_table(win, cols), _qlike_table(m16, cols)
    rows = []
    for col in cols:
        A = Qw.groupby(["ticker", "year"])[col].mean().unstack()
        A = A[A.notna().sum(1) >= 2]                                    # бумага с < 2 годами цели в окне не входит (как в crit_panel)
        pens = pd.DataFrame([_pen({y: v for y, v in r.items() if np.isfinite(v)}) for _, r in A.iterrows()])
        npen = len(pens.columns)
        for y in SEL_YEARS:
            rows.append({"series": col, "year": y, "role": "выбор", "n_tickers": len(A), "ann_qlike": float(A[y].mean()), "pen": float(pens[y].mean()) if y in pens else np.nan,
                         "crit_part": float(pens[y].mean() / npen) if y in pens else np.nan})
        D = Qa[Qa["year"].isin(list(DESC_YEARS))].groupby(["ticker", "year"])[col].mean().unstack()
        rows += [{"series": col, "year": y, "role": "описательно", "n_tickers": int(D[y].notna().sum()), "ann_qlike": float(D[y].mean()), "pen": np.nan, "crit_part": np.nan} for y in D.columns]
    df = pd.DataFrame(rows)
    for ref in ("k1", "d2"):
        r = df[df["series"] == ref].set_index("year")["ann_qlike"]
        df["d_ann_" + ref] = df["ann_qlike"] - df["year"].map(r)
    return df


def crit_from_years(yt: pd.DataFrame) -> pd.Series:
    """Тождество: сумма crit_part по годам 2018…2021 = crit_panel."""
    return yt.groupby("series")["crit_part"].sum(min_count=1)
