"""8. Дни остановки торгов: у тикера нет бара — доходность через паузу целиком на последнем баре
до неё (позиция держится), портфель дня паузы — среднее по торгующимся тикерам, без нулей.
"""
import numpy as np
import pandas as pd

from conftest import read_dev
from fastbt import data as fd
from fastbt import msp

D = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-11"])
CA = np.array([100.0, 101, 102, 103, 104])
CB = np.array([50.0, 51, np.nan, np.nan, 49])                 # B стоит 03-03…03-10


def _fill_zero(f):
    def g(pos, *a, **k):
        full = pd.MultiIndex.from_product([sorted(pos["date"].unique()), sorted(pos["ticker"].unique()),
                                           sorted(pos["strategy"].unique())],
                                          names=["date", "ticker", "strategy"])
        p = pos.set_index(["date", "ticker", "strategy"]).reindex(full, fill_value=0.0).reset_index()
        return f(p, *a, **k)
    return g


def _drop_gap(f):
    def g(*a, **k):
        td = f(*a, **k)
        gap = np.r_[np.diff(td.dates).astype("timedelta64[D]").astype(int), 1] > 4
        td.log_ret[gap] = 0.0
        return td
    return g


MUTATIONS = {"halt_fill_zero": lambda mp: mp.setattr(msp, "portfolio", _fill_zero(msp.portfolio)),
             "halt_drop_return": lambda mp: mp.setattr(fd, "load_ticker", _drop_gap(fd.load_ticker))}


def _bars(monkeypatch):
    rows = []
    for t, c in (("AAA", CA), ("BBB", CB)):
        for d, x in zip(D, c):
            if np.isfinite(x):
                rows.append(dict(ticker=t, date=d, open=x, high=x, low=x, close=x, volume=1.0))
    df = pd.DataFrame(rows)
    monkeypatch.setattr(fd, "daily_df", lambda: df)
    monkeypatch.setattr(fd, "vpred_df", lambda: pd.DataFrame(
        {"date": D, "ticker": "BBB", "sigma_h1": 0.02}))
    fd.load_ticker.cache_clear()


def test_return_across_halt(monkeypatch):
    _bars(monkeypatch)
    td = fd.load_ticker("BBB")
    fd.load_ticker.cache_clear()
    assert len(td.dates) == 3
    assert np.isclose(td.log_ret[1], np.log(49 / 51))               # 03-02 → 03-11 одним баром


def test_portfolio_on_halt_days(monkeypatch):
    _bars(monkeypatch)
    rows = []
    for t in ("AAA", "BBB"):
        td = fd.load_ticker(t)
        for d, lr in zip(td.dates, td.log_ret):
            rows.append(dict(date=pd.Timestamp(d), ticker=t, strategy="S1", position=1.0,
                             daily_gross_return=lr))
    fd.load_ticker.cache_clear()
    p = msp.portfolio(pd.DataFrame(rows), pd.Series([0.0], pd.to_datetime(["2020-01-01"])), 0.0)
    la = np.log(CA[1:] / CA[:-1])
    assert np.isclose(p.loc["2021-03-02", "r"], (la[1] + np.log(49 / 51)) / 2)
    assert np.isclose(p.loc["2021-03-03", "r"], la[2])                # только AAA
    assert np.isclose(p.loc["2021-03-03", "exposure"], 1.0)
    assert np.isclose(p.loc["2021-03-02", "turnover"], 0.0)          # пауза — не оборот


def test_dev_years_have_no_halts():
    """Наблюдение на данных ≤ 2021: в годах разработки 2018–2021 торгуют все 17 тикеров каждый день
    (пропуски — только 2014–2015), то есть ветка «пауза» в таблицах разработки не включается."""
    d = read_dev(fd.OHLCV, ["ticker", "date"])
    d = d[d["date"] >= "2017-01-01"]
    assert (d.groupby("date")["ticker"].nunique() == 17).all()
