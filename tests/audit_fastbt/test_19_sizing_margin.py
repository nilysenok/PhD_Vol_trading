"""19. Размер позиции, маржа, плечо: единичные позиции, потолок плеча D, ставка и плата за плечо.

Позиции A/B/C/D-движков ∈ {−1, 0, +1} (100% номинала, без
плеча); непрерывный D — min(cap, (σ*/σ)^γ); МСП: под ставку идёт 1 − доля·экспозиция за
календарные дни [d, d_next); при экспозиции > 1 — плата (ключевая + 2%)/365 за занятое сверх
капитала (спецификация п. 10; сценарий маржи 20% — решение владельца 26.10, 00:55).

Пути: stats.pos_from_trades, execlayer.exec_layer, approach_cd.d_cont_scales, msp.portfolio,
s9.Engine.port/_finish, margin.margin_table.
"""
import numpy as np
import pandas as pd
import pytest

from _eng import mutant
from fastbt import approach_cd, grids, margin, msp, s9
from fastbt.config import Config
from fastbt.signals.interface import TickerData

RATE = 0.10


def _share_ignored(f):
    def g(*a, **k):
        k["cap_share"] = 1.0
        return f(*a, **k)
    return g


MUTATIONS = {
    "lev_no_cap": lambda mp: mp.setattr(approach_cd, "d_cont_scales", mutant(
        approach_cd.d_cont_scales, [("np.minimum((tv / s) ** g, cap)", "(tv / s) ** g", 1)])),
    "margin_share_ignored": lambda mp: mp.setattr(msp, "portfolio", _share_ignored(msp.portfolio)),
    "margin_cost_dropped": lambda mp: mp.setattr(s9.Engine, "_finish", mutant(
        s9.Engine._finish, [("total = r + fdr - mg", "total = r + fdr", 1)])),
    # Н4 вернули: плата за плечо за один день
    "margin_one_day": lambda mp: (
        mp.setattr(s9.Engine, "_finish", mutant(s9.Engine._finish, [
            ("/ 365.0 * self.cal", "/ 365.0", 1)])),
        mp.setattr(msp, "portfolio", mutant(msp.portfolio, [("/ 365.0 * cal", "/ 365.0", 1)]))),
}


def test_engine_positions_unit_size():
    from _eng import bars, run_lvl
    from fastbt import execlayer, stats
    c, h, lo = bars(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300)))
    rng = np.random.default_rng(4)
    L = sorted(rng.choice(300, 40, replace=False))
    S = [t for t in sorted(rng.choice(300, 40, replace=False)) if t not in L]
    tr = run_lvl(c, h, lo, longs=L, shorts=S, a1=1.5, a2=3, mh=20, cd=5)
    te, tx, td = (np.array([t[i] for t in tr], dt) for i, dt in
                  ((0, np.int64), (1, np.int64), (2, np.int8)))
    pos = stats.pos_from_trades(te, tx, td, len(tr), len(c))
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0} and len(tr) > 5
    ex = execlayer.exec_layer(pos, np.ones(len(c)), 2, 5, 3, 0.0, len(c))
    assert set(np.unique(ex)) <= {-1.0, 0.0, 1.0}


def test_d_continuous_size_capped():
    n = 50
    sg = np.full(n, 0.002)               # σ мала → (σ*/σ)^γ ≫ cap
    sg[10:20] = 0.5
    td = TickerData("ZZZ", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"),
                    *([np.ones(n)] * 5), np.zeros(n), {h: sg for h in ("h1", "h5", "h22")})
    rows, params = approach_cd.d_cont_scales(td, True)
    caps = np.array([p["cap"] for p in params])
    assert set(caps) == set(grids.D_MAX_LEVS)
    assert (rows.max(axis=1) <= caps + 1e-12).all()
    assert (np.isclose(rows.max(axis=1), caps)).sum() > len(caps) // 2   # потолок достигается
    k = params.index(dict(target_vol=grids.D_TARGET_VOLS[0], cap=grids.D_MAX_LEVS[0],
                          gamma=grids.D_CONT_GAMMA[0], horizon=grids.D_HORIZONS[0], continuous=True))
    tv, g = grids.D_TARGET_VOLS[0], grids.D_CONT_GAMMA[0]
    assert np.isclose(rows[k, 15], min(grids.D_MAX_LEVS[0], (tv / (0.5 * np.sqrt(252))) ** g))


# ---------- МСП: свободный капитал и маржа ----------

DATES = pd.to_datetime(["2021-03-04", "2021-03-05", "2021-03-09", "2021-03-10"])  # чт, пт, вт, ср
CAL = np.array([1.0, 4.0, 1.0, 1.0])     # [d, d_next): пт → вт (8 марта — выходной); последняя — 1


def _pos():
    p1, p2 = [1, 1, 0, 0], [0, -1, -1, 1]
    rows = []
    for tk, p in (("AAA", p1), ("BBB", p2)):
        for d, x in zip(DATES, p):
            rows.append(dict(date=d, ticker=tk, strategy="S1", position=float(x),
                             daily_gross_return=float(x) * 0.01))
    return pd.DataFrame(rows)


def _hand(share, comm=0.0):
    exp_ = np.array([0.5, 1.0, 0.5, 0.5])
    turn = np.array([[1, 0, 1, 0], [0, 1, 0, 2]], float)        # |Δpos| по парам
    gross = np.array([[1, 1, 0, 0], [0, -1, -1, 1]], float) * 0.01
    r = (gross - turn * comm).mean(axis=0)
    rf = RATE / 365
    return r + np.maximum(0, 1 - share * exp_) * rf * CAL, exp_


@pytest.mark.parametrize("share", [1.0, 0.2])
def test_msp_free_capital_by_hand(share):
    rate = pd.Series([RATE], pd.to_datetime(["2020-01-01"]))
    p = msp.portfolio(_pos(), rate, 0.001, cap_share=share)
    tot, exp_ = _hand(share, 0.001)
    assert np.allclose(p["exposure"], exp_)
    assert np.allclose(p["cal_days"], CAL)
    assert np.allclose(p["total"], tot) and (p["margin"] == 0).all()
    assert np.allclose(p["excess"], tot - RATE / 365 * CAL)


def test_margin_table_scenarios(monkeypatch):
    monkeypatch.setattr(msp, "rate_cbr", lambda: pd.Series([RATE], pd.to_datetime(["2020-01-01"])))
    pos = pd.concat([_pos().assign(approach=a, test_year=2021) for a in ("A", "C")])
    cfg = Config(msp_comm=0.0)
    t = margin.margin_table(pos, cfg, [2021], ("2021-01-01", "2021-12-31"), 20).set_index(
        ["scenario", "approach"])
    for scen, share in margin.SCEN.items():
        tot, exp_ = _hand(share)
        r = t.loc[(scen, "A")]
        assert np.isclose(r["total_pct_year"], tot.mean() * 252 * 100)
        assert np.isclose(r["capital_in_positions_pct"], exp_.mean() * share * 100)
    assert set(margin.SCEN.values()) == {1.0, 0.2}


def _engine(dates, kr):
    n = len(dates)
    e = object.__new__(s9.Engine)
    e.r = {"date": np.arange(n)}
    e.first = np.r_[True, np.zeros(n - 1, bool)]
    e.lr_row, e.cnt = np.zeros(n), np.ones(n)
    e.dates = pd.DatetimeIndex(dates)
    e.kr = np.full(n, kr)
    e.rf = e.kr / 365.0
    e.cal = msp.cal_days(e.dates)
    return e


def test_leverage_financing_weekday():
    """Плечо 1,6 (вес B 1,6 при позиции 1) в споте: плата 0,6·(ключевая + 2%)/365 за день,
    свободного капитала нет; в сценарии маржи 20% занято 0,32 — платы нет."""
    d = pd.to_datetime(["2021-03-02", "2021-03-03", "2021-03-04"])            # вт, ср, чт
    e = _engine(d, RATE)
    pos = np.ones(3)
    p = s9.Engine.port(e, pos, 0.0, 1.0, np.full(3, 1.6))
    assert np.allclose(p["exposure"], 1.6)
    assert np.allclose(p["total"], -0.6 * (RATE + 0.02) / 365)
    pm = s9.Engine.port(e, pos, 0.0, 0.2, np.full(3, 1.6))
    assert np.allclose(pm["total"], (1 - 0.32) * RATE / 365)


def test_leverage_financing_calendar_days():
    """Н4 (исправлено): плата за плечо — за календарные дни [d, d_next), как ставка."""
    d = pd.to_datetime(["2021-03-04", "2021-03-05", "2021-03-09"])            # чт, пт → вт
    e = _engine(d, RATE)
    p = s9.Engine.port(e, np.ones(3), 0.0, 1.0, np.full(3, 1.6))
    assert np.allclose(p["total"], -0.6 * (RATE + 0.02) / 365 * np.array([1.0, 4.0, 1.0]))
    pos = _pos()
    pos["position"] *= 3.0                                                     # D с плечом до 3
    pos["daily_gross_return"] = 0.0
    q = msp.portfolio(pos, pd.Series([RATE], pd.to_datetime(["2020-01-01"])), 0.0)
    used = np.array([1.5, 3.0, 1.5, 1.5])
    assert np.allclose(q["margin"], (used - 1) * (RATE + 0.02) / 365 * CAL)
