"""7. Корпоративные события: цены до даты события × factor, объём ÷ factor (data/corporate_actions.csv);
доходность через событие — рыночная, без скачка от смены номинала.
"""
import numpy as np
import pandas as pd

from conftest import bdays
from fastbt import data as fd

MUTATIONS = {"corp_skip": lambda mp: mp.setattr(fd, "corp_actions",
                                                lambda t: fd.corp_actions(t).iloc[0:0])}
_orig_corp = fd.corp_actions


def _synthetic(monkeypatch):
    d = bdays("2021-03-01", 6)
    close = np.array([10.0, 10.2, 10.1, 1.02, 1.03, 1.01])      # сплит 1:10 с бара 3
    df = pd.DataFrame({"ticker": "ZZZ", "date": pd.to_datetime(d), "open": close,
                       "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": [100.0, 100, 100, 1000, 1000, 1000]})
    ev = pd.DataFrame({"ticker": ["ZZZ"], "date": [pd.Timestamp(d[3])], "factor": [0.1],
                       "status": ["applied"]})
    monkeypatch.setattr(fd, "daily_df", lambda: df)
    monkeypatch.setattr(fd, "vpred_df", lambda: pd.DataFrame(
        {"date": pd.to_datetime(d), "ticker": "ZZZ", "sigma_h1": 0.02}))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: ev)           # data/corporate_actions.csv
    fd.load_ticker.cache_clear()
    return close


def test_split_by_hand(monkeypatch):
    close = _synthetic(monkeypatch)
    td = fd.load_ticker("ZZZ", True)
    fd.load_ticker.cache_clear()
    adj = np.r_[close[:3] * 0.1, close[3:]]
    assert np.allclose(td.close, adj)
    assert np.allclose(td.volume, [1000.0] * 6)
    assert np.isclose(td.log_ret[2], np.log(1.02 / 1.01))           # через событие — рынок
    assert np.abs(td.log_ret).max() < 0.05


def test_irao_2015_consolidation_dev(dev_data):
    """Данные ≤ 2021: IRAO 18.12.2014 → 20.01.2015 (×100) после корректировки — без скачка."""
    raw = dev_data.load_ticker("IRAO", False)
    adj = dev_data.load_ticker("IRAO", True)
    i = int(np.searchsorted(raw.dates, np.datetime64("2015-01-20"))) - 1
    assert raw.log_ret[i] > 4.0
    assert abs(adj.log_ret[i]) < 0.15
    assert np.allclose(raw.log_ret[i + 1:], adj.log_ret[i + 1:])


def test_no_unexplained_jumps_dev(dev_data):
    """≤ 2021, после корректировок: |лог-доходность| > 0,35 только у известных рыночных
    движений MTLR осени 2014 (спецификация п. 1)."""
    known = {("MTLR", "2014-09-22"), ("MTLR", "2014-12-18")}
    bad = []
    for t in fd.TICKERS:
        td = dev_data.load_ticker(t, True)
        lr = td.log_ret[:-1]                                   # последний бар 2021 — без пары
        for i in np.where(np.abs(lr) > 0.35)[0]:
            k = (t, str(pd.Timestamp(td.dates[i]).date()))
            if k not in known:
                bad.append(k)
    assert not bad, bad
