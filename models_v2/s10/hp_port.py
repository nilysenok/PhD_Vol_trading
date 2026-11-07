"""P0, G5: переносимость замороженных контролей K2 (XGB x2_best_j = Б-2 + F4), K3 (LGBM C0_inh_F4), K6 (HAR-J + E8) на каждую бумагу (R10).
Входы, найденные чтением кода: models_v2/s5/b2/feat_day.py (day_stats: бары open/high/low/close/volume основной сессии → rq, medrv, park, range2, hhi, ln volume; cross_stats: бары IMOEX),
models_v2/s4/mkt.py (load_papers: бары 16 бумаг PAPERS для рыночного блока Б-2), s9/hv2_mkt + hv2_sync (E8: синхронные с IMOEX бары), история для годового LOYO 2015…Y−1 и 252 пар HAR-J.
RVI и stress_p3 в входах Б-2/F4/C0_inh_F4 не найдены (grep по s5/b2, s6/xgb, s6/lgbm: stress — только при Tm.freq == "adapt", не у C0_inh_F4 и x2_best_j).
Правило переносимости (задано в P0 до счёта, не подбиралось): бары имеют все колонки; high >= low > 0 и volume > 0 на >= 99 % дней; первый канонический день <= 2014-12-31 и >= 126 дней до 2015-01-01
(история для LOYO 2015); общих баров с IMOEX на >= 95 % дней; >= 252 пар HAR-J к 2017-01-03; бары 16 бумаг PAPERS есть."""
from __future__ import annotations

import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P

COLS = ("begin", "open", "high", "low", "close", "volume")
PAPERS = ("AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX", "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "TATN", "VTBR")
HIST_BEFORE, MIN_DAYS_BEFORE = pd.Timestamp("2014-12-31"), 126


def facts(t: str, bars: pd.DataFrame, d: pd.DataFrame) -> dict:
    m = D.main_session(bars)
    day = m["begin"].dt.normalize()
    ok_px = ((m["high"] >= m["low"]) & (m["low"] > 0)).groupby(day).all().reindex(d.index)
    vol = (m.groupby(day)["volume"].sum() > 0).reindex(d.index)
    return {"ticker": t, "has_cols": all(c in bars.columns for c in COLS), "share_days_px_ok": float(ok_px.mean()), "share_days_vol_pos": float(vol.mean()),
            "first_day": str(d.index.min().date()), "n_days_before_2015": int((d.index <= HIST_BEFORE).sum()),
            "share_days_sync_imoex": float(d["cov"].notna().mean()) if "cov" in d else float("nan"),
            "n_pairs_at_20170103": int((d.index < pd.Timestamp(P.FIRST_TARGET)).sum()) - 1}


def verdict(f: dict, papers_ok: bool) -> dict:
    why = []
    if not f["has_cols"]:
        why.append("нет колонок open/high/low/close/volume")
    if f["share_days_px_ok"] < 0.99 or f["share_days_vol_pos"] < 0.99:
        why.append("high/low/volume вырождены на >1 %% дней (px %.3f, vol %.3f)" % (f["share_days_px_ok"], f["share_days_vol_pos"]))
    hist = f["n_days_before_2015"] >= MIN_DAYS_BEFORE
    if not hist:
        why.append("история до 2015-01-01: %d дней < %d (годовой LOYO 2015 невозможен)" % (f["n_days_before_2015"], MIN_DAYS_BEFORE))
    sync = f["share_days_sync_imoex"] >= 0.95
    if not sync:
        why.append("общих баров с IMOEX %.3f < 0.95 дней" % f["share_days_sync_imoex"])
    if f["n_pairs_at_20170103"] < P.MIN_PAIRS:
        why.append("пар HAR-J к 2017-01-03: %d < %d" % (f["n_pairs_at_20170103"], P.MIN_PAIRS))
    if not papers_ok:
        why.append("нет баров бумаг PAPERS")
    ml = not why
    k6 = f["has_cols"] and sync and f["n_pairs_at_20170103"] >= P.MIN_PAIRS
    return {"K2_portable": ml, "K3_portable": ml, "K6_portable": bool(k6), "reason_not_portable": "; ".join(why) if why else "",
            "note": "код K2/K3 жёстко на SBER (prov.BARS) — в P1 нужна обёртка; здесь проверены только входы"}


def table(bars_by: dict, daily_by: dict) -> pd.DataFrame:
    papers_ok = all(P.bars_path(p).exists() for p in PAPERS)
    rows = []
    for t in P.TICKERS:
        f = facts(t, bars_by[t], daily_by[t])
        rows.append({**f, **verdict(f, papers_ok)})
    return pd.DataFrame(rows)
