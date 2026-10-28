"""Л-1 (Ж-073): синтетика с известным ответом для fastbt/inv_tikery.py + подложенные ошибки. Данные не читает."""
import numpy as np
import pandas as pd
import pytest

from fastbt import inv_tikery as it
from fastbt.exec_e_types import Trade


class FakeEng:
    """Минимум от Engine: даты и _finish; excess = r (ставку и плечо не моделируем)."""
    dates = pd.date_range("2018-01-01", periods=4)

    def _finish(self, r, gross, exp_, turn, cap):
        return pd.DataFrame(dict(excess=r, gross=gross, exposure=exp_, turnover=turn), index=self.dates)


def _tr(side, d0, b0, d1, b1):
    return Trade("X", d0, d0, b0, 100.0, d1, b1, 101.0, "vremya", side, 0.00035, 0.00035)


CUM = np.array([0, 100, 200, 300, 400])        # 100 баров в дне


def _check_duration():
    t = _tr(1, 1, 30, 3, 10)                   # день 1 бар 30 → день 3 бар 10: 70 + 100 + 11 = ... позиции 130 и 310
    rec = it.trade_record(t, CUM, 2018, 0.01)
    assert rec["bary"] == 180 and rec["dni"] == 2


def _check_sides():
    recs = [dict(side=1, pnl=0.02), dict(side=-1, pnl=-0.01), dict(side=-1, pnl=0.03)]
    assert it.po_storonam(recs) == (1, 2, 0.02, pytest.approx(0.02))
    assert it.po_storonam(recs)[3] == pytest.approx(0.02) and it.po_storonam(recs)[2] == 0.02


def _check_net():
    di, g, sd = np.array([0, 1, 1]), np.array([0.01, 0.02, 0.0]), np.array([1.0, 1.0, 1.0])
    ex = it.standalone_excess(FakeEng(), di, g, sd, np.ones(3), it.COMM)
    # день 1: g − Σ sides·0,035 % = 0,02 − 2·0,00035
    assert ex.iloc[1] == pytest.approx(0.02 - 2 * 0.00035, abs=1e-15) and ex.iloc[0] == pytest.approx(0.01 - 0.00035)
    assert ex.iloc[2] == 0.0 and ex.iloc[3] == 0.0


def test_komissiya_0035():
    assert it.COMM == 0.00035 and it.COMM_REF == 0.0004
    _check_net()


def test_dlitelnost_bary_i_dni():
    _check_duration()
    t = _tr(1, 2, 99, 3, 0)                    # через границу дня: 1 бар
    assert it.trade_record(t, CUM, 2018, 0.0)["bary"] == 1


def test_storony_long_short_razdeleny():
    _check_sides()


def test_mutatsiya_komissiya_004_lovitsya(monkeypatch):
    monkeypatch.setattr(it, "COMM", 0.0004)
    with pytest.raises(AssertionError):
        _check_net()


def test_mutatsiya_dni_vmesto_bar_lovitsya(monkeypatch):
    monkeypatch.setattr(it, "bars_between", lambda cum, d0, b0, d1, b1: d1 - d0)
    with pytest.raises(AssertionError):
        _check_duration()


def test_mutatsiya_short_v_longe_lovitsya(monkeypatch):
    real = it.po_storonam

    def bad(recs):
        return real([{**r, "side": 1} if r["side"] < 0 else r for r in recs])
    monkeypatch.setattr(it, "po_storonam", bad)
    with pytest.raises(AssertionError):
        _check_sides()


def test_ticker_row_i_g2():
    ra = [dict(side=1, pnl=0.02, year=2018, bary=10, dni=1), dict(side=-1, pnl=-0.01, year=2019, bary=30, dni=3)]
    rb = [dict(side=-1, pnl=0.05, year=2019, bary=20, dni=2)]
    a, b = it.ticker_row("A", ra, [0.5, 1.5], 4, 2), it.ticker_row("B", rb, [float("nan")] * 2, 1, 1)
    assert a[3:5] == [1, 1] and a[7] == 0.5 and a[8] == pytest.approx(0.5) and a[9] == 20 and a[10] == 2
    assert a[11] == 0.5 and a[12] == 0.5 and a[13] == pytest.approx(2.0) and a[14] == pytest.approx(-1.0)
    itog = it.ticker_row("ITOG", ra + rb, [0.0, 0.0], 5, 3)
    it.check_g2([a, b], itog)
    itog[15] += 1e-6                            # потеряли PnL: G2 обязан падать
    with pytest.raises(SystemExit):
        it.check_g2([a, b], itog)


def test_msr_years_pustaya_lenta_nan():
    ex = pd.Series(np.zeros(5), index=pd.date_range("2018-01-01", periods=5))
    assert all(np.isnan(x) for x in it.msr_years(pd.concat([ex, ex.set_axis(pd.date_range("2019-01-01", periods=5))])))
