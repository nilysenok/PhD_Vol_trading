"""Проверка функции потерь Б-2: QLIKE-оптимальный сдвиг масштаба (PIT) и правило эпсилон. Диагностика, не trial."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import post
from models_v2.s5 import common, pit

NOTE = "диагностика, не trial бюджета"
TRAIN_LOSS = {"objective": "reg:squarederror (MSE)", "target": "ln RV_{t+1} − ln RV_m,t", "back_transform": "σ̂² = exp(ln RV_m + f + s²/2), s² — MSE блока остановки",
              "link_to_qlike": "MSE по ln-цели с поправкой s²/2 даёт несмещённость по среднему RV только при логнормальной ошибке; QLIKE этого не требует"}


def calibrated(df: pd.DataFrame, extra_closed: int = 0) -> dict:
    """{модель: (c_k, σ̂²·e^{c_k})} — c_k по закрытым прогнозам (target_date <= d_k, >= 252)."""
    out = {}
    for m in ("b2", "harj"):
        c = pit.scale_shift(df["rv"], df["s2_" + m], df["d"].to_numpy(), df["target_date"].to_numpy(), extra_closed=extra_closed)
        out[m] = (c, df["s2_" + m].to_numpy() * np.exp(c))
    return out


def table(df: pd.DataFrame) -> pd.DataFrame:
    cal = calibrated(df)
    active = post.closed_count(df["d"].to_numpy(), df["target_date"].to_numpy()) >= pit.MIN_CLOSED
    masks = dict(common.period_masks(df["target_date"]))
    masks["all_after_252_closed"] = active
    rows = []
    for m in ("b2", "harj"):
        c, s2c = cal[m]
        for variant, s2 in (("raw", df["s2_" + m].to_numpy()), ("scale_shift_pit", s2c)):
            for lab, mk in masks.items():
                if mk.sum() == 0:
                    continue
                q = metrics.qlike(df["rv"].to_numpy()[mk], s2[mk])
                rows.append({"model": m, "variant": variant, "slice": lab, "n": int(mk.sum()), "qlike": float(q.mean()),
                             "mean_r": float((df["rv"].to_numpy()[mk] / s2[mk]).mean()), "mean_c": float(c[mk].mean()), "note": NOTE})
    return pd.DataFrame(rows)


def eps_check(df: pd.DataFrame, dl: pd.DataFrame) -> pd.DataFrame:
    eps = pd.Series(pit.eps_floor(dl["rv"].to_numpy(float)), index=dl.index).reindex(df["d"]).to_numpy()
    r = {"n": len(df), "rule": "eps_d = 0.1 * min(дневная RV по строкам <= d)", "min_rv_target": float(df["rv"].min()),
         "min_s2_b2": float(df["s2_b2"].min()), "min_s2_harj": float(df["s2_harj"].min()), "eps_min": float(eps.min()), "eps_max": float(eps.max()),
         "n_rv_below_eps": int((df["rv"].to_numpy() < eps).sum()), "n_s2_b2_below_eps": int((df["s2_b2"].to_numpy() < eps).sum()),
         "n_s2_harj_below_eps": int((df["s2_harj"].to_numpy() < eps).sum()), "rv_all_positive": bool((df["rv"] > 0).all()),
         "pilot_floor_applied": False}
    return pd.DataFrame([r])
