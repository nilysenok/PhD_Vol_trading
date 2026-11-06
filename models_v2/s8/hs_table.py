"""Ж-212, этап A: ежедневная таблица ERROR_STRUCTURE_DAILY.parquet (2271 строка). Все признаки строки d известны к 18:50 d (PIT);
цель rv — RV дня target_date = d+1. Поля past_* — только по закрытым прогнозам (target_date <= d), `hs_pit`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily, metrics
from models_v2.s5 import pit
from models_v2.s6.e import comp
from models_v2.s7 import hf_data as D
from models_v2.s8 import hs_paths as P
from models_v2.s8 import hs_pit

PAST = (("eH", "e_HAR"), ("eX", "e_XJ"), ("eL", "e_LF"), ("dqX", "dq_HAR_XJ"), ("dqL", "dq_HAR_LF"))


def pit_columns(o: pd.DataFrame, extra: int = 0) -> pd.DataFrame:
    """past_<имя>_last / _m5 / _m22 / _m63 по закрытым прогнозам. extra>0 — подложенная ошибка («закрыто + extra»)."""
    o = o.copy()
    d, td = o["d"].to_numpy("datetime64[ns]"), o["target_date"].to_numpy("datetime64[ns]")
    for nm, col in PAST:
        x = o[col].to_numpy(float)
        o["past_%s_last" % nm] = hs_pit.past_last(x, d, td, extra)
        for w in (5, 22, 63):
            o["past_%s_m%d" % (nm, w)] = hs_pit.past_mean(x, d, td, w, extra=extra)
    return o


def base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Потери, ошибки e = ln RV − ln σ̂², расхождения прогнозов (известны на d: оба прогноза сделаны на d)."""
    o = df[["d", "target_date", "rv", "HAR", "XJ", "LF", "st_g4", "st_rv"]].copy()
    for m in ("HAR", "XJ", "LF"):
        o["e_" + m] = np.log(o["rv"] / o[m])
        o["q_" + m] = metrics.qlike(o["rv"], o[m])
    o["dq_HAR_XJ"], o["dq_HAR_LF"], o["dq_XJ_LF"] = o["q_HAR"] - o["q_XJ"], o["q_HAR"] - o["q_LF"], o["q_XJ"] - o["q_LF"]
    o["delta_HX"], o["delta_HL"], o["delta_XL"] = (np.log(o[a] / o[b]) for a, b in (("HAR", "XJ"), ("HAR", "LF"), ("XJ", "LF")))
    return o


def har_columns(o: pd.DataFrame, F, g, fg: pd.DataFrame) -> pd.DataFrame:
    """Компоненты HAR-J (ln RV_d/w/m, ln(1+J)), полувариации и ВСЕ признаки утверждённого набора F1–F6 (feats_b2)."""
    X, eps = F.X[g], 1e-12
    for j, nm in ((0, "ln_rv_d"), (1, "ln_rv_w"), (2, "ln_rv_m"), (3, "ln1p_J"), (7, "rs_plus"), (8, "rs_minus")):
        o[nm] = X[:, j]
    o["rs_asym"] = np.log((o["rs_minus"] + eps) / (o["rs_plus"] + eps))
    o["ln_rvw_m"] = o["ln_rv_w"] - o["ln_rv_m"]
    for c in fg.columns:
        o["F_" + c] = fg[c].to_numpy(float)
    o["ter_rv_w"] = D.ter_w(F, g)
    return o


def market_columns(o: pd.DataFrame) -> pd.DataFrame:
    """RV рынка (IMOEX, 10-минутные бары до 18:40, те же правила, что для SBER), её терциль по 252 предыдущим и режим RVI (d vs медиана до d)."""
    idly, _ = daily.build_daily(pd.read_parquet(P.IMOEX), index_days=None)
    for nm in ("rv", "rv_w", "rv_m"):
        o["mkt_" + nm] = idly[nm].reindex(pd.DatetimeIndex(o["d"])).to_numpy(float)
    o["mkt_ter_w"] = np.nan_to_num(pit.past_bucket(o["mkt_rv_w"].to_numpy(float), [1 / 3, 2 / 3]), nan=-1).astype(int)
    rvi = pd.read_parquet("%s/data/fastbt/s9/rvi_1850.parquet" % P.ROOT)
    rvi = rvi.reset_index() if "date" not in rvi.columns else rvi
    o["rvi_regime"] = metrics.rvi_regime(o["d"], rvi).fillna("na").to_numpy()
    return o


def build(w: dict | None = None) -> pd.DataFrame:
    w = w or D.world()
    o = base_columns(w["df"])
    o = har_columns(o, w["F"], w["g"], w["fg"])
    o = market_columns(o)
    return pit_columns(o)


def save(o: pd.DataFrame) -> None:
    P.OUT.mkdir(parents=True, exist_ok=True)
    o.to_parquet(P.OUT / "ERROR_STRUCTURE_DAILY.parquet")
