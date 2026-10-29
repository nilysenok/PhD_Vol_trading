"""Расследование 29.10: почему падает fastbt/tests/test_s56_equiv.py.

Причина: legacy-контекст S5/S6 (fastbt/signals/s56.py::_legacy_ctx) берёт pivot/VWAP из СЫРОГО
daily_df(), а td.close/high/low/open — с обратной поправкой на дивиденды (dividendy.adjust_ohlc, Ж-059 п. 2).
Тикеры без дивидендов (MTLR) совпадают, с дивидендами (SBER) — нет. Тест фиксирует причину, правок кода нет.
"""
import pytest

from fastbt import data
from fastbt.indicators import compute_base
from fastbt.signals import s56
from fastbt.signals.interface import check_family, get_family


@pytest.fixture
def bez_popravki(monkeypatch):
    monkeypatch.setattr(data.dividendy, "adjust_ohlc", lambda t, d, c, *bars: tuple(b.copy() for b in bars))
    data.load_ticker.cache_clear()
    s56._CACHE.clear()
    yield
    data.load_ticker.cache_clear()
    s56._CACHE.clear()


def _bad(sid, tk):
    td = data.load_ticker(tk)
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    return check_family(get_family(sid), td, ind)["bad"]


def test_td_close_otlichaetsya_ot_syrogo_u_sber_ne_u_mtlr():
    d = data.daily_df()
    for tk, differs in (("SBER", True), ("MTLR", False)):
        raw = d[d["ticker"] == tk].sort_values("date")["close"].values
        assert (not (data.load_ticker(tk).close == raw).all()) is differs


def test_s5_rashoditsya_s_legacy_iz_za_popravki_dividendov():
    s56._CACHE.clear()
    assert _bad("S5", "SBER") != []        # воспроизведение симптома


def test_bez_popravki_dividendov_s5_s6_sovpadayut(bez_popravki):
    assert _bad("S5", "SBER") == []
    assert _bad("S6", "SBER") == []
