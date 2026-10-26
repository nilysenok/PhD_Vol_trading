"""3. Свободный капитал под фактическую ставку ЦБ и та же ставка в избыточной доходности.

total = r + max(0, 1 − cap·exp)·rf·cal − маржа, excess = total − rf·cal (тот же rf, те же дни);
rf — ключевая ставка, действующая на дату (решения ЦБ 2018–2021 по датам вступления в силу).
"""
import numpy as np
import pandas as pd
import pytest

from conftest import KEY_RATE, read_dev
from fastbt import msp, s9

# ключевая ставка ЦБ, дата вступления в силу → %, 2018–2021 (пресс-релизы Банка России)
CBR = [("2017-12-18", 7.75), ("2018-02-12", 7.50), ("2018-03-26", 7.25), ("2018-09-17", 7.50),
       ("2018-12-17", 7.75), ("2019-06-17", 7.50), ("2019-07-29", 7.25), ("2019-09-09", 7.00),
       ("2019-10-28", 6.50), ("2019-12-16", 6.25), ("2020-02-10", 6.00), ("2020-04-27", 5.50),
       ("2020-06-22", 4.50), ("2020-07-27", 4.25), ("2021-03-22", 4.50), ("2021-04-26", 5.00),
       ("2021-06-15", 5.50), ("2021-07-26", 6.50), ("2021-09-13", 6.75), ("2021-10-25", 7.50),
       ("2021-12-20", 8.50)]
# найдено в данных (не в движке): ряд key_rate_cbr сдвинут на 1 рабочий день у двух решений
KNOWN_DATA_DEV = {pd.Timestamp("2018-02-09"), pd.Timestamp("2019-10-28")}


def _shift_month(f):
    return lambda: f().pipe(lambda s: s.set_axis(s.index + pd.Timedelta(days=30)))


def _excess_const(f):
    def g(*a, **k):
        p = f(*a, **k)
        p["excess"] = p["total"] - 0.16 / 365 * p["cal_days"]
        return p
    return g


def _finish_const(f):
    def g(self, *a, **k):
        p = f(self, *a, **k)
        p["excess"] = p["total"] - 0.16 / 365 * self.cal
        return p
    return g


MUTATIONS = {
    "rate_shift_month": lambda mp: mp.setattr(msp, "rate_cbr", _shift_month(msp.rate_cbr)),
    "rate_const16": lambda mp: mp.setattr(msp, "rate_cbr",
                                          lambda: pd.Series(0.16, pd.to_datetime(["2014-01-01"]))),
    "excess_other_rate": lambda mp: (mp.setattr(msp, "portfolio", _excess_const(msp.portfolio)),
                                     mp.setattr(s9.Engine, "_finish",
                                                _finish_const(s9.Engine._finish))),
}


def _pos(dates, position, gross):
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "ZZZ", "strategy": "S1",
                         "position": position, "daily_gross_return": gross})


def test_engine_rate_matches_cbr_2018_2021(monkeypatch, tmp_path):
    p = tmp_path / "key_rate_dev.parquet"
    read_dev(KEY_RATE).to_parquet(p)                  # копия ряда ≤ 2021
    monkeypatch.setattr(msp, "KEY_RATE", p)
    days = pd.bdate_range("2018-01-01", "2021-12-31")
    port = msp.portfolio(_pos(days, 0.0, 0.0), msp.rate_cbr(), 0.0)
    got = (port["fdr"] / port["cal_days"] * 365).values        # экспозиция 0 → fdr = rf·cal
    sch = pd.Series([v / 100 for _, v in CBR], pd.to_datetime([d for d, _ in CBR]))
    want = sch.reindex(sch.index.union(days)).ffill().reindex(days).values
    bad = set(days[~np.isclose(got, want)])
    assert bad <= KNOWN_DATA_DEV, sorted(bad)[:5]


def test_free_capital_and_excess_by_hand():
    d = pd.to_datetime(["2021-03-18", "2021-03-19", "2021-03-22", "2021-03-23"])
    rate = pd.Series([0.0425, 0.045], pd.to_datetime(["2021-03-01", "2021-03-22"]))
    port = msp.portfolio(_pos(d, [0.0, 1.0, 0.5, 0.0], [0.0, 0.01, -0.005, 0.0]), rate, 0.0)
    rf = np.array([0.0425, 0.0425, 0.045, 0.045]) / 365
    cal = port["cal_days"].values
    e = np.array([0.0, 1.0, 0.5, 0.0])
    assert np.allclose(port["fdr"], (1 - e) * rf * cal)
    assert np.allclose(port["total"], [0.0, 0.01, -0.005, 0.0] + (1 - e) * rf * cal)
    assert np.allclose(port["total"] - port["excess"], rf * cal)       # та же ставка, те же дни
    # маржа 20%: под ставку 1 − 0,2·exp
    pm = msp.portfolio(_pos(d, [0.0, 1.0, 0.5, 0.0], 0.0), rate, 0.0, cap_share=0.2)
    assert np.allclose(pm["fdr"], (1 - 0.2 * e) * rf * cal)


def _engine(dates, rate):
    e = object.__new__(s9.Engine)
    e.dates = pd.DatetimeIndex(dates)
    kr = rate.reindex(rate.index.union(e.dates)).sort_index().ffill().bfill().reindex(e.dates)
    e.kr = kr.values
    e.rf = e.kr / 365
    cal = np.ones(len(dates))
    cal[1:] = np.diff(e.dates.values).astype("timedelta64[D]").astype(float)
    e.cal = cal
    return e


def test_s9_engine_same_rate_in_excess():
    d = pd.to_datetime(["2021-03-18", "2021-03-19", "2021-03-22", "2021-03-23"])
    rate = pd.Series([0.0425, 0.045], pd.to_datetime(["2021-03-01", "2021-03-22"]))
    e = _engine(d, rate)
    exp_ = np.array([0.0, 1.0, 0.5, 0.0])
    r = np.array([0.0, 0.01, -0.005, 0.0])
    p = e._finish(r, r, exp_, np.zeros(4), 1.0)
    assert np.allclose(p["total"], r + (1 - exp_) * e.rf * e.cal)
    assert np.allclose(p["total"] - p["excess"], e.rf * e.cal)
    assert np.allclose(e.rf * 365, [0.0425, 0.0425, 0.045, 0.045])


def test_s9_forbids_back_rate():
    """Ъ ~23:20: режим «back» (rate_restored, дни (d_prev, d]) — только печати v1; Engine С9
    (разработка и тест v2) его не принимает."""
    import inspect
    src = inspect.getsource(s9.Engine.__init__)
    assert 'rate.attrs.get("cal") != "back"' in src and "assert" in src


def test_cash_accrues_over_position_period():
    """Н2 исправлено: позиция пятницы держится пт→пн — кэш пятницы за 3 дня по экспозиции и
    ставке пятницы; выходные перед сменой ставки (пн 22.03.2021) идут по старой ставке."""
    d = pd.to_datetime(["2021-03-18", "2021-03-19", "2021-03-22", "2021-03-23"])
    rate = pd.Series([0.0425, 0.045], pd.to_datetime(["2021-03-01", "2021-03-22"]))
    port = msp.portfolio(_pos(d, [0.0, 1.0, 0.0, 0.0], 0.0), rate, 0.0)
    rf = np.array([0.0425, 0.0425, 0.045, 0.045]) / 365
    assert np.allclose(port["cal_days"], [1, 3, 1, 1])
    assert np.allclose(port["fdr"], [rf[0], 0.0, rf[2], rf[3]])        # пт: экспозиция 1
    assert np.allclose(port["total"] - port["excess"], rf * [1, 3, 1, 1])
    assert np.allclose(msp.cal_days(d), [1, 3, 1, 1])
    assert np.allclose(msp.cal_days(d, forward=False), [1, 1, 3, 1])   # ворота (печати)
    import inspect
    assert "msp.cal_days(self.dates)" in inspect.getsource(s9.Engine.__init__)


def test_gate_restored_rate_keeps_print_convention():
    r = pd.Series([0.1], pd.to_datetime(["2021-01-01"]))
    r.attrs["cal"] = "back"
    d = pd.to_datetime(["2021-03-18", "2021-03-19", "2021-03-22"])
    assert np.allclose(msp.portfolio(_pos(d, 0.0, 0.0), r, 0.0)["cal_days"], [1, 1, 3])
