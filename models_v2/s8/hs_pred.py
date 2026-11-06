"""Ж-212, этап A, вопрос 3: какие признаки (известные к 18:50 d) предсказывают относительное преимущество dq = QLIKE(HAR) − QLIKE(XJ) на d+1.
OLS dq ~ a + b·z (z — стандартизованный признак; стандартизация по выборке — диагностика, не прогноз), HAC Ньюи–Уэст лаг 5; Холм по числу признаков;
устойчивость: знак b в P1 2017–2021 / P2 2022 / P3 2023–2025 и t без 24.02.2022; вариант с dq, подрезанным на 1 %/99 % (весь эффект 2022 — в хвосте)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s8 import hs_diag as H
from models_v2.s8 import hs_enc

EXTRA = ("ln_rv_d", "ln_rv_w", "ln_rv_m", "ln1p_J", "rs_asym", "ln_rvw_m", "delta_HX", "delta_XL", "delta_HL", "mkt_ln_rv", "mkt_ln_rv_w", "mkt_ln_rv_m")


def feature_names(o: pd.DataFrame) -> list:
    return [c for c in o.columns if c.startswith("F_") or c.startswith("past_") or c in EXTRA]


def prepare(o: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    for a, b in (("mkt_ln_rv", "mkt_rv"), ("mkt_ln_rv_w", "mkt_rv_w"), ("mkt_ln_rv_m", "mkt_rv_m")):
        o[a] = np.log(o[b])
    return o


def _one(o: pd.DataFrame, f: str, y: np.ndarray, yw: np.ndarray, mk: dict) -> dict:
    z = o[f].to_numpy(float)
    ok = np.isfinite(z)
    if ok.sum() < 300 or np.nanstd(z) == 0:
        return {"feature": f, "n": int(ok.sum())}
    z = (z - np.nanmean(z)) / np.nanstd(z)
    r = hs_enc.mz_hac(y[ok], z[ok])
    row = {"feature": f, "n": int(ok.sum()), "b": r["b"], "t_nw": r["t_b"], "p": r["p_b"], "r2": r["r2"], "t_nw_winsor": hs_enc.mz_hac(yw[ok], z[ok])["t_b"]}
    k = ok & mk["no_240222"]
    row["t_nw_ex_240222"] = hs_enc.mz_hac(y[k], z[k])["t_b"]
    for p in ("P1_2017_2021", "P2_2022", "P3_2023_2025"):
        kk = ok & mk[p]
        row["b_" + p] = hs_enc.mz_hac(y[kk], z[kk])["b"] if kk.sum() > 50 else np.nan
    return row


def table(o: pd.DataFrame) -> pd.DataFrame:
    o = prepare(o)
    y = o["dq_HAR_XJ"].to_numpy(float)
    lo, hi = np.quantile(y, [0.01, 0.99])
    yw, mk = np.clip(y, lo, hi), H.period_masks(o)
    t = pd.DataFrame([_one(o, f, y, yw, mk) for f in feature_names(o)])
    t = t[t["p"].notna()].copy()
    t["p_holm"] = dmod.holm(t["p"].to_numpy())
    s = np.sign(t[["b_P1_2017_2021", "b_P2_2022", "b_P3_2023_2025"]].to_numpy())
    t["sign_same_in_2of3"] = (s == np.sign(t["b"].to_numpy())[:, None]).sum(axis=1) >= 2
    t["robust"] = (t["p_holm"] < 0.10) & t["sign_same_in_2of3"] & (np.abs(t["t_nw_ex_240222"]) > 2) & (np.abs(t["t_nw_winsor"]) > 2)
    return t.reindex(t["t_nw"].abs().sort_values(ascending=False).index).reset_index(drop=True)
