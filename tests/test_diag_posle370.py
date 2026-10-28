"""Повтор диагностики после #370 (Ж-058): суммы PnL «после» равны сумме журнала на синтетике с известным ответом (D·0,87),
«до» сверяется с журналом #366. Мутации: дивиденд не копится на сделке; PnL «до» выдан за «после»; сверка «до» пропускает расхождение."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import diag_posle370 as dp
from fastbt import exec_e
from fastbt.exec_e_types import Day, Params

DATES = np.array(["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"], dtype="datetime64[D]")
START = 0


def day(o, h, l, c, n=3):
    return Day(*[np.full(n, float(x)) for x in (o, h, l, c)])


def _trades(sig, d1, d2c=101.0, div=None, accrue=True):
    days = [day(100, 100.5, 99.5, 100), d1, day(100, 102, 96, d2c), day(100, 100.5, 99.5, 100)]
    args = ("AAA", days, np.array(sig), np.full(4, 0.02), Params(k=1.0, tick=0.1, comm=0.0, exit_signal=True))
    if accrue:
        with dp.accrue_dividends():
            return exec_e.run(*args, div=div).trades
    return exec_e.run(*args, div=div).trades


DIV = np.array([0.0, 0.0, 2.0, 0.0])
FILLED = day(100, 100.5, 97.9, 99)          # лонг: вход по L = 98, выход close 101


def test_posle_sum_equals_known_journal_sum_long():
    rows = dp.trade_rows(_trades([1, 1, 0, 0], FILLED, div=DIV), DATES, START, True)
    assert len(rows) == 1 and sum(r[7] for r in rows) == pytest.approx((101 + 2 * 0.87) / 98 - 1, abs=1e-6)
    assert rows[0][9] == pytest.approx(2 * 0.87 / 98, abs=1e-6)


def test_posle_short_pays_dividend():
    rows = dp.trade_rows(_trades([-1, -1, 0, 0], day(100, 102.5, 99, 99), d2c=97.0, div=DIV), DATES, START, True)   # вход L = 102
    assert len(rows) == 1 and sum(r[7] for r in rows) == pytest.approx(-((97 + 2 * 0.87) / 102 - 1), abs=1e-6)


def test_before_pnl_has_no_dividend_and_differs_from_after():
    tr = _trades([1, 1, 0, 0], FILLED, div=DIV)
    before = sum(r[7] for r in dp.trade_rows(tr, DATES, START, False))
    after = sum(r[7] for r in dp.trade_rows(tr, DATES, START, True))
    assert before == pytest.approx(101 / 98 - 1, abs=1e-6) and after - before == pytest.approx(2 * 0.87 / 98, abs=1e-6)


def test_open_trade_and_warmup_excluded():
    tr = _trades([1, 1, 0, 0], FILLED, div=DIV)
    assert dp.trade_rows(tr, DATES, 3, True) == []                    # вход раньше старта окна — не входит


def test_mutation_dividend_not_accrued_caught():
    rows = dp.trade_rows(_trades([1, 1, 0, 0], FILLED, div=DIV, accrue=False), DATES, START, True)
    with pytest.raises(AssertionError):
        assert sum(r[7] for r in rows) == pytest.approx((101 + 2 * 0.87) / 98 - 1, abs=1e-6)


def test_mutation_before_passed_as_after_caught():
    rows = dp.trade_rows(_trades([1, 1, 0, 0], FILLED, div=DIV), DATES, START, False)
    with pytest.raises(AssertionError):
        assert sum(r[7] for r in rows) == pytest.approx((101 + 2 * 0.87) / 98 - 1, abs=1e-6)


def test_reconcile_before_rejects_mismatch(monkeypatch):
    rows = [["", "", "A", "long", "1", "1", "vremya", 0.01, "2018", 0.0]] * 2
    monkeypatch.setattr(dp, "old_journal", lambda sid, suf: [0.01, 0.01])
    assert dp.reconcile_before("S1", "", rows) == (2, pytest.approx(0.02))
    monkeypatch.setattr(dp, "old_journal", lambda sid, suf: [0.01, 0.02])      # Σ расходится
    with pytest.raises(SystemExit):
        dp.reconcile_before("S1", "", rows)
    monkeypatch.setattr(dp, "old_journal", lambda sid, suf: [0.01])            # число сделок расходится
    with pytest.raises(SystemExit):
        dp.reconcile_before("S1", "", rows)


def test_msr_by_year_splits_years():
    import pandas as pd
    idx = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04", "2019-01-02", "2019-01-03", "2019-01-04"])
    ex = pd.Series([0.01, 0.02, -0.005, -0.01, -0.02, 0.0], index=idx)
    m = dp.msr_by_year(ex)
    assert m[2018] > 0 > m[2019]
