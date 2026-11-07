"""P0, G6: исследовательская панель panel_base.parquet — строка (бумага i, дата t): цель RV бумаги в её СЛЕДУЮЩИЙ СОБСТВЕННЫЙ день t⁺ (R2), K1 σ̂² на t⁺ (прогноз из дня t), QLIKE K1,
рыночная RV, J, части дня, корреляция, семиковариации, признаки X/M/C (сырые и z по прошлому бумаги). Колонки прогнозов K2–K7 и гибридов дописываются в листах P1–P2."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s10 import hp_feat as F, hp_paths as P, hp_sync as SY

DAY_COLS = {"rv": "rv_t", "rv_w": "rv_w_t", "rv_m": "rv_m_t", "j": "j", "rv_open": "rv_open", "rv_mid": "rv_mid", "rv_close": "rv_close", "rs_pos": "rs_pos", "rs_neg": "rs_neg",
            "corr": "corr_d", "beta": "beta_d", "cov_pp": "cov_pp", "cov_nn": "cov_nn", "cov_mix": "cov_mix", "idio": "idio_d"}


def cross_frames(daily: dict) -> tuple:
    """Матрицы дата × бумага (NaN — у бумаги нет дня t): ln RV_d/RV_m и 1/0 «RV_d > RV_w». Без заполнения (R4)."""
    lnrel = SY.wide({t: F.ln(d["rv"]) - F.ln(d["rv_m"]) for t, d in daily.items()})
    up = SY.wide({t: (d["rv"] > d["rv_w"]).astype(float).where(d["rv_w"].notna()) for t, d in daily.items()})
    return lnrel, up


def features_all(daily: dict, imo: pd.DataFrame, k1: dict) -> dict:
    """{бумага: таблица X/M/C на её датах (сырые + z)}."""
    lnrel, up = cross_frames(daily)
    cs = SY.cross_stats(lnrel, up)
    rank, dev = SY.cross_rank(lnrel, cs["m_med"])
    out = {}
    for t, d in daily.items():
        f = pd.concat([F.individual(d, k1[t]), F.market_own(d, imo), F.structural(d)], axis=1)
        for c in ("m_med", "m_disp", "m_share_up"):
            f[c] = cs[c].reindex(d.index)
        f["c_rank"], f["c_dev_med"] = rank[t].reindex(d.index), dev[t].reindex(d.index)
        out[t] = F.add_z(f[F.X_COLS + F.M_COLS + F.C_COLS])
    return out


def ticker_rows(t: str, d: pd.DataFrame, k: pd.DataFrame, f: pd.DataFrame, imo_rv: pd.Series) -> pd.DataFrame:
    """Строки панели бумаги t: основа — таблица K1 (дата t = d, t⁺ = следующий собственный день)."""
    r = pd.DataFrame({"ticker": t, "group": P.group(t), "sector": P.SECTOR[t], "date": k["d"].to_numpy(), "t_plus": k["target_date"].to_numpy(),
                      "gap": k["gap"].to_numpy(), "gap_days": k["gap_days"].to_numpy(), "cross_seg": k["cross_seg"].to_numpy(),
                      "rv_tplus": k["rv"].to_numpy(), "sigma2_k1": k["sigma2"].to_numpy(), "n_pairs_k1": k["n_pairs"].to_numpy()})
    r["qlike_k1"] = metrics.qlike(r["rv_tplus"], r["sigma2_k1"])
    ix = pd.DatetimeIndex(r["date"])
    dd = d.reindex(ix)
    for src, dst in DAY_COLS.items():
        r[dst] = dd[src].to_numpy()
    r["mkt_rv_t"] = imo_rv.reindex(ix).to_numpy()
    r["mkt_rv_tplus"] = imo_rv.reindex(pd.DatetimeIndex(r["t_plus"])).to_numpy()
    return pd.concat([r.reset_index(drop=True), f.reindex(ix).reset_index(drop=True)], axis=1)


def build(daily: dict, imo: pd.DataFrame, k1: dict) -> pd.DataFrame:
    feats = features_all(daily, imo, k1)
    parts = [ticker_rows(t, daily[t], k1[t], feats[t], imo["rv"]) for t in P.TICKERS]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def check(panel: pd.DataFrame, daily: dict) -> list:
    """Ошибки панели (пусто = ок): 28 бумаг; t⁺ > t и t⁺ — следующий собственный день (по дневному ряду бумаги); gap = (t⁺ − t > 5 сут.); ключ (бумага, t) уникален; QLIKE конечен."""
    bad = []
    if panel["ticker"].nunique() != 28:
        bad.append("бумаг %d ≠ 28" % panel["ticker"].nunique())
    if not (panel["t_plus"] > panel["date"]).all():
        bad.append("t⁺ <= t")
    if panel.duplicated(["ticker", "date"]).any():
        bad.append("дубли (бумага, дата)")
    if not (panel["gap"] == ((panel["t_plus"] - panel["date"]).dt.days > P.GAP_DAYS)).all():
        bad.append("gap не по правилу > 5 сут.")
    if not np.isfinite(panel["qlike_k1"]).all():
        bad.append("QLIKE K1 нефинитен")
    for t, g in panel.groupby("ticker"):
        ix = daily[t].index
        if not (ix.get_indexer(pd.DatetimeIndex(g["t_plus"])) == ix.get_indexer(pd.DatetimeIndex(g["date"])) + 1).all():
            bad.append("%s: t⁺ не следующая строка собственного ряда бумаги" % t)
    return bad


def checkpoint() -> dict:
    """Продолжение docs/otchety/harj_hybrid_v2/CHECKPOINT.json (R7): N_registry_start 585, budget 24 (10/10/4), used 0, leaf P0."""
    return {"program": "Ж-215/Ж-217 HAR-J Panel Hybrid, h=1", "continues": "docs/otchety/harj_hybrid_v2/CHECKPOINT.json", "N_registry_start": 585,
            "budget": {"total": 24, "xgb": 10, "lgbm": 10, "reserve": 4}, "used": 0, "leaf": "P0", "updated": time.strftime("%Y-%m-%d %H:%M")}


def write_checkpoint() -> None:
    P.REP.mkdir(parents=True, exist_ok=True)
    (P.REP / "CHECKPOINT.json").write_text(json.dumps(checkpoint(), ensure_ascii=False, indent=1))
