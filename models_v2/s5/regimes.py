"""REGIME_DIAGNOSTICS: режимы только по прошлому (квинтили RV_w по 252 предыдущим, стресс Г4, переходы)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import stats


def _rows(df: pd.DataFrame, mask, kind: str, label: str) -> list:
    if mask.sum() == 0:
        return []
    sub = df[mask]
    ws = stats.win_share(sub["qlike_b2"], sub["qlike_harj"])
    out = []
    for name in ("b2", "harj"):
        r = stats.full_metrics(sub["rv"], sub["s2_" + name])
        out.append({"kind": kind, "label": label, "model": name, "n": r["n"], "qlike": r["qlike"], "mean_r": r["mean_r"],
                    "bias_ln": r["bias_ln"], "rmse_ln": r["rmse_ln"], "win_share_b2": ws})
    return out


def transitions(q: pd.Series) -> pd.Series:
    prev = q.shift(1)
    lab = pd.Series(np.where(prev.isna() | q.isna(), None, "q" + prev.fillna(-1).astype(int).astype(str) + ">q" + q.fillna(-1).astype(int).astype(str)),
                    index=q.index, dtype=object)
    return lab


def direction(q: pd.Series) -> pd.Series:
    d = q - q.shift(1)
    return pd.Series(np.where(d.isna(), None, np.where(d > 0, "up", np.where(d < 0, "down", "same"))), index=q.index, dtype=object)


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = df["quintile_w"]
    for g in range(5):
        rows += _rows(df, (q == g).to_numpy(), "quintile_w", "q%d" % g)
    st = df["stress_g4"].astype(bool)
    rows += _rows(df, st.to_numpy(), "stress_g4", "stress") + _rows(df, (~st).to_numpy(), "stress_g4", "no_stress")
    rows += _rows(df, df["mar_jun_2020"].to_numpy(bool), "window", "mar_jun_2020")
    rows += _rows(df, df["y2022"].to_numpy(bool), "window", "y2022")
    tr, di = transitions(q), direction(q)
    for lab in sorted(x for x in tr.dropna().unique()):
        rows += _rows(df, (tr == lab).to_numpy(), "transition", lab)
    for lab in ("up", "down", "same"):
        rows += _rows(df, (di == lab).to_numpy(), "transition_dir", lab)
    return pd.DataFrame(rows)
