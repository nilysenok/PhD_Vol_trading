"""P0, G1: инвентарь 28 бумаг, календарь дата × бумаги, расписание сессии по годам, хэши входов. Читает только бары (без test_locked/)."""
from __future__ import annotations

import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s9 import hv2_sched as S

STOP = (pd.Timestamp("2022-02-24"), pd.Timestamp("2022-03-24"))   # 25.02–23.03.2022 — закрытие биржи; границы с запасом на день до/после


def turnover_by_year(bars: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    """Медианный дневной оборот основной сессии (млн руб.) по годам, по каноническим дням бумаги."""
    m = D.main_session(bars)
    v = m.groupby(m["begin"].dt.normalize())["value"].sum().reindex(days).dropna() / 1e6
    return {"to_med_%d_mln" % y: float(g.median()) for y, g in v.groupby(v.index.year)}


def gap_pairs(days: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(days[1:] - days[:-1]).dt.days


def one(t: str, bars: pd.DataFrame, d: pd.DataFrame, imo_days: pd.DatetimeIndex) -> dict:
    days = d.index
    gp = gap_pairs(days)
    inside = imo_days[(imo_days >= days.min()) & (imo_days <= days.max())]
    sched = S.day_schedule(D.main_session(bars))
    in_stop = days[(days >= STOP[0]) & (days <= STOP[1])]
    return {"ticker": t, "group": P.group(t), "sector": P.SECTOR[t],
            "sector_basis": "ручной справочник: общеизвестная отрасль эмитента (без внешней сверки); не признак",
            "first_date": str(days.min().date()), "last_date": str(days.max().date()), "n_days": len(days),
            "n_holes_vs_imoex": int((~inside.isin(days)).sum()), "n_gap_pairs_gt5": int((gp > P.GAP_DAYS).sum()), "max_gap_days": int(gp.max()),
            "n_days_stop_window_2022": len(in_stop), "n_days_sched_not_ok": int((~sched["ok"].reindex(days).fillna(False)).sum()),
            "n_days_bars_lt52": int((d["n_bars"] < 52).sum()), **turnover_by_year(bars, days)}


def calendar(days: dict) -> pd.DataFrame:
    """Дата × бумага: 1 — у бумаги есть каноническая торговая строка, 0 — нет (по объединению дат)."""
    idx = sorted(set().union(*[set(x) for x in days.values()]))
    return pd.DataFrame({t: pd.Index(idx).isin(x).astype(int) for t, x in days.items()}, index=pd.DatetimeIndex(idx, name="date")).reset_index()


def schedule(bars_by: dict) -> pd.DataFrame:
    rows = []
    for t, b in bars_by.items():
        r = S.regime_periods(S.day_schedule(D.main_session(b)))
        rows.append(r.assign(ticker=t))
    return pd.concat(rows, ignore_index=True)[["ticker", "year", "regime", "days", "first_day", "last_day", "n_med", "n_min", "ok_share"]]


def hashes() -> pd.DataFrame:
    fs = [P.bars_path(t) for t in P.TICKERS + ["IMOEX"]] + [P.COMP, P.M10_SBER]
    return pd.DataFrame({"file": [str(f.relative_to(P.ROOT)) for f in fs], "sha256_16": [P.sha16(f) for f in fs]})


def check_inventory(inv: pd.DataFrame, cal: pd.DataFrame) -> list:
    """Ошибки инвентаря (пусто = ок): 28 бумаг, без дублей, календарь содержит все, SBER-хэш b451cb48473ac15c, IMOEX-хэш cadc3e41cda11533."""
    bad = []
    if sorted(inv["ticker"]) != P.TICKERS or len(inv) != 28:
        bad.append("состав бумаг ≠ 28 из R1")
    if set(P.TICKERS) - set(cal.columns):
        bad.append("календарь без части бумаг")
    h = hashes().set_index("file")["sha256_16"]
    for f, e in (("data/iss_check/bars_10m_full/SBER.parquet", "b451cb48473ac15c"), ("data/iss_check/index_10m/IMOEX.parquet", "cadc3e41cda11533"),
                 ("data/regen/xgb_lgbm_full_v1/e/components.parquet", "0b18be29aff34f8e")):
        if h.get(f) != e:
            bad.append("хэш %s: %s ≠ %s" % (f, h.get(f), e))
    return bad


def run(bars_by: dict, cores: dict) -> tuple:
    imo_days = cores["IMOEX"].index
    inv = pd.DataFrame([one(t, bars_by[t], cores[t], imo_days) for t in P.TICKERS])
    return inv, calendar({t: cores[t].index for t in P.TICKERS + ["IMOEX"]})
