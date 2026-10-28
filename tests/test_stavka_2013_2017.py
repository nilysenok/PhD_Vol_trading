"""Ставка ЦБ 2013–2017 в `msp.rate_cbr()` (PR данных для В1): исследовательский ряд до 2017-12-18,
ряд владельца после; стык совпадает; до 2013-09-17 — явная ошибка (Д4); ключ кэша видит оба файла (Д5).
Мутации — только в памяти (monkeypatch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastbt import msp  # noqa: E402
from fastbt.etap1 import E1Inputs  # noqa: E402

T = pd.Timestamp
FIRST, SEAM = T("2013-09-17"), T("2017-12-18")


def test_value_2016_06_01_is_11pct():
    assert msp.rate_cbr().asof(T("2016-06-01")) == pytest.approx(0.11)


def test_dates_of_entry_2015_2017_table():
    """Таблица «дата вступления → ставка» 2015–2017 (cbr.ru hd_base)."""
    exp = {"2015-02-02": 15.0, "2015-03-16": 14.0, "2015-05-05": 12.5, "2015-06-16": 11.5, "2015-08-03": 11.0,
           "2016-06-14": 10.5, "2016-09-19": 10.0, "2017-03-27": 9.75, "2017-05-02": 9.25, "2017-06-19": 9.0,
           "2017-09-18": 8.5, "2017-10-30": 8.25, "2017-12-18": 7.75}
    r = msp.rate_cbr()
    for d, v in exp.items():
        assert r.loc[T(d)] == pytest.approx(v / 100), d
        assert r.asof(T(d) - pd.Timedelta(days=1)) != pytest.approx(v / 100), f"{d}: ставка не меняется в день вступления"


def test_seam_2017_12_18_equal_and_7_75():
    o, r = msp._read_rate(msp.RATE_OWNER), msp._read_rate(msp.RATE_RESEARCH)
    assert o.loc[SEAM] == pytest.approx(0.0775) and r.loc[SEAM] == pytest.approx(0.0775)
    s = msp.rate_cbr()
    assert s.index.is_unique and s.index.is_monotonic_increasing and s.loc[SEAM] == pytest.approx(0.0775)


def test_seam_mismatch_is_error(monkeypatch, tmp_path):
    """Мутация: исследовательский ряд на стыке расходится с владельцем — ошибка, не молчаливая склейка."""
    bad = tmp_path / "r.csv"
    bad.write_text(msp.RATE_RESEARCH.read_text().replace("2017-12-18,7.75", "2017-12-18,7.50"))
    monkeypatch.setattr(msp, "RATE_RESEARCH", bad)
    with pytest.raises(ValueError, match="стык"):
        msp.rate_cbr()


def test_seam_taken_from_research_instead_of_owner_is_caught(monkeypatch):
    """Мутация № «стык из исследовательского ряда»: после 2017-12-18 должен быть ряд владельца
    (7,50 с 2018-02-12). Склейка, обрезающая владельца по стыку, теряет строки — тест это видит."""
    r = msp.rate_cbr()
    assert r.asof(T("2018-02-12")) == pytest.approx(0.075)
    o = msp._read_rate(msp.RATE_OWNER)
    post = r[(r.index >= SEAM) & (r.index < T("2022-01-01"))]
    assert post.equals(o[(o.index >= SEAM) & (o.index < T("2022-01-01"))])
    research_only = msp._read_rate(msp.RATE_RESEARCH)
    assert r[r.index < T("2022-01-01")].index.max() > research_only.index.max(), "ряд обрезан по исследовательскому"


def test_before_first_date_is_explicit_error_and_no_bfill():
    with pytest.raises(ValueError, match="ряд начинается"):
        msp.rate_on(pd.date_range("2013-09-16", periods=3))
    ok = msp.rate_on(pd.date_range(FIRST, periods=3))
    assert (ok == 0.055).all()


def test_mutation16_bfill_constant_is_caught():
    """Мутация № 16: ставка до начала ряда вытянута назад константой — константа длиннее любого интервала."""
    r = msp.rate_cbr()
    days = pd.date_range("2013-01-01", "2013-09-30")
    x = r.reindex(r.index.union(days)).sort_index().ffill().bfill().reindex(days)
    assert x.loc[:"2013-09-16"].eq(0.055).all()          # bfill выдумал бы 5,5 % до введения
    with pytest.raises(ValueError):
        msp.rate_on(days)                               # настоящий путь отказывает


def test_no_empty_days_2015_2021():
    n = int(msp.rate_on(pd.date_range("2015-01-01", "2021-12-31")).isna().sum())
    assert n == 0
    bd = pd.bdate_range("2015-01-01", "2021-12-31")
    assert int(msp.rate_on(bd).isna().sum()) == 0


def test_golden_2018_2021_unchanged():
    """2018–2021 побайтно как у ряда владельца (без исследовательских строк)."""
    o = msp._read_rate(msp.RATE_OWNER)
    r = msp.rate_cbr()
    a = r[(r.index >= T("2018-01-01")) & (r.index < T("2022-01-01"))]
    b = o[(o.index >= T("2018-01-01")) & (o.index < T("2022-01-01"))]
    assert a.to_numpy().tobytes() == b.to_numpy().tobytes() and (a.index == b.index).all()


def _digest(rate: pd.Series) -> str:
    return E1Inputs(sk=None, sigma={}, bars={}, close_n6={}, rezhim={}, sobytiya={}, tick={}, rate=rate).digest()


def test_cache_key_tracks_research_file(monkeypatch, tmp_path):
    """Д5: подмена значения в исследовательском файле меняет ключ кэша входов движка."""
    base = _digest(msp.rate_cbr())
    alt = tmp_path / "r.csv"
    alt.write_text(msp.RATE_RESEARCH.read_text().replace("2015-08-03,11.00", "2015-08-03,11.25"))
    monkeypatch.setattr(msp, "RATE_RESEARCH", alt)
    assert _digest(msp.rate_cbr()) != base


def test_cache_key_mutation_research_dropped_is_caught():
    """Мутация «хэш без исследовательского файла»: ряд только владельца даёт другой ключ, чем полный."""
    full = _digest(msp.rate_cbr())
    o = msp._read_rate(msp.RATE_OWNER)
    only_owner = _digest(o[o.index < T("2022-01-01")])
    assert full != only_owner
