"""Группы по величине RV (по факту — объяснительно; по известной на d — PIT) и события скачков / резких изменений."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import pit, stats

LABELS = ("very_low", "low", "mid", "high", "extreme")
INNER = (0.2, 0.4, 0.8, 0.95)
OFFSETS = range(-5, 6)


def fact_bucket(rv) -> np.ndarray:
    """Группы по квантилям всей выборки (знает будущее — только для объяснения)."""
    x = np.asarray(rv, float)
    return (x[:, None] > np.quantile(x, INNER)[None, :]).sum(axis=1).astype(float)


def pit_bucket(df: pd.DataFrame, dl: pd.DataFrame, mutate_future: bool = False) -> np.ndarray:
    """Группы RV_d относительно 252 предыдущих значений дневного ряда. mutate_future=True — мутация: окно с текущим днём."""
    b = pd.Series(pit.past_bucket(dl["rv"].to_numpy(float), INNER, include_current=mutate_future), index=dl.index)
    return b.reindex(df["d"]).to_numpy()


def table(df: pd.DataFrame, bucket, basis: str, note: str) -> pd.DataFrame:
    rows = []
    for g, lab in enumerate(LABELS):
        m = bucket == g
        if m.sum() == 0:
            continue
        sub = df[m]
        ws = stats.win_share(sub["qlike_b2"], sub["qlike_harj"])
        for name in ("b2", "harj"):
            r = stats.full_metrics(sub["rv"], sub["s2_" + name])
            rows.append({"basis": basis, "note": note, "group": lab, "model": name, "n": r["n"], "qlike": r["qlike"], "mean_r": r["mean_r"],
                         "bias_ln": r["bias_ln"], "rmse_ln": r["rmse_ln"], "under_share": float((sub["s2_" + name] < sub["rv"]).mean()),
                         "win_share_b2": ws})
    return pd.DataFrame(rows)


def event_flags(dl: pd.DataFrame, kind: str) -> np.ndarray:
    """Событие на дне дневного ряда: kind 'jump' — J/RV выше 0,9-квантиля 252 предыдущих значений; 'rv_change' — то же для Δ ln RV."""
    if kind == "jump":
        x = (dl["j"] / dl["rv"]).to_numpy(float)
    else:
        x = np.concatenate([[np.nan], np.diff(np.log(dl["rv"].to_numpy(float)))])
    thr = pit.q_prior_nan(x, 0.9, pit.WIN)
    with np.errstate(invalid="ignore"):
        return x > thr


def events(df: pd.DataFrame, dl: pd.DataFrame, kind: str) -> pd.DataFrame:
    idx = pit.decluster(event_flags(dl, kind))
    pos = pd.Series(np.arange(len(df)), index=df["d"])
    arr = {c: df[c].to_numpy() for c in ("rv", "s2_b2", "s2_harj", "qlike_b2", "qlike_harj", "err_ln_b2", "err_ln_harj")}
    rows = []
    for k in OFFSETS:
        acc = []
        for i in idx:
            if 0 <= i + k < len(dl) and dl.index[i + k] in pos.index:
                acc.append(int(pos[dl.index[i + k]]))
        a = np.array(acc, int)
        rec = {"kind": kind, "offset": k, "n_events": len(a)}
        if len(a):
            rec.update(ln_rv_target=np.log(arr["rv"][a]).mean(), ln_s2_b2=np.log(arr["s2_b2"][a]).mean(), ln_s2_harj=np.log(arr["s2_harj"][a]).mean(),
                       qlike_b2=arr["qlike_b2"][a].mean(), qlike_harj=arr["qlike_harj"][a].mean(), bias_ln_b2=arr["err_ln_b2"][a].mean(),
                       bias_ln_harj=arr["err_ln_harj"][a].mean())
        rows.append(rec)
    return pd.DataFrame(rows)
