"""Регрессия (расследование 29.10, починка): эталон S5/S6 и td — на ОДНОМ ряду.

Причина красного fastbt/tests/test_s56_equiv.py была в том, что `_legacy_ctx` строил pivot/VWAP из сырого
daily_df(), а td — со скорректированным на дивиденды рядом (Ж-059 п. 2, #370, mutations.md № 33).
Теперь эталон берёт OHLC из td. Мутация: эталон построен по сырому ряду -> тест обязан падать.
"""
import dataclasses

import numpy as np
import pytest

from fastbt import data
from fastbt.legacy import load_legacy
from fastbt.signals import s56


@pytest.fixture(autouse=True)
def _cache():
    s56._CACHE.clear()
    yield
    s56._CACHE.clear()


def _raw(td):
    d = data.daily_df()
    x = d[d["ticker"] == td.ticker].sort_values("date")
    return dataclasses.replace(td, **{k: x[k].values.astype(np.float64) for k in ("open", "high", "low", "close")})


def _pivot_gap(td_ref, td_sig):
    """max |P эталона (на ряду td_ref) − P ядра на ряду td_sig|."""
    pc = s56._legacy_ctx(load_legacy(), td_ref)[1]
    P = pc[("weekly", "classic")][0]
    mine = s56._pivots(s56._gid(td_sig.dates, "weekly"), td_sig.high, td_sig.low, td_sig.close, False)[0]
    return np.nanmax(np.abs(P - mine))


def test_ryad_sber_otlichaetsya_ot_syrogo():
    td = data.load_ticker("SBER")
    assert not (td.close == _raw(td).close).all()      # поправка на дивиденды действует


def test_etalon_na_odnom_ryadu_s_td():
    td = data.load_ticker("SBER")
    assert _pivot_gap(td, td) == 0.0


def test_mutatsiya_etalon_na_syrom_ryadu_lovitsya():
    td = data.load_ticker("SBER")
    assert _pivot_gap(_raw(td), td) > 0.0               # эталон по сырому ряду расходится с ядром
