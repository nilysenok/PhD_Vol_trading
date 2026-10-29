"""Прямая дивидендная корректировка (Ж-137 п. 7, Д3, PR div-pryamaya): ряд на день d знает только выплаты с отсечкой <= d.
Синтетика, данные не читаются. python3 -m pytest tests/test_dividendy_pryamaya.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy

N = 400
DATES = pd.bdate_range("2018-01-01", periods=N).values
EV = [("T", "2018-06-15", 3.0), ("T", "2019-03-14", 5.0)]       # last_cum_date; день гэпа — следующий рабочий


def _table(tmp_path, monkeypatch, rows):
    cal = pd.bdate_range("2005-01-03", "2035-12-31").values.astype("datetime64[ns]")
    monkeypatch.setattr(dividendy, "_calendar", lambda: cal)
    f = tmp_path / "div.csv"
    rec = lambda c: (pd.Timestamp(c) + pd.offsets.BDay(2)).date()          # noqa: E731
    f.write_text("ticker,record_date,last_cum_date,announce_date,dps_rub\n"
                 + "".join(f"{t},{rec(c)},{c},{c},{d}\n" for t, c, d in rows))
    monkeypatch.setattr(dividendy, "TABLE", f)


def _price():
    return 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.01, N)))


def obratnaya(ticker, dates, close, *bars):
    """Прежняя (обратная) корректировка — эталон доходностей и мутация «корректировка обратная»."""
    div = dividendy.gap_divs(ticker, dates)
    f = np.ones(len(dates))
    for g in np.flatnonzero(div):
        f[:g] *= 1.0 - div[g] / close[g - 1]
    return tuple(np.asarray(b, float) * f for b in bars)


def _prefiks_ok(adj, tmp_path, monkeypatch):
    """Ряд, обрезанный по cut, с таблицей только известных событий == префикс полного ряда ПОБАЙТНО."""
    c = _price()
    _table(tmp_path, monkeypatch, EV)
    full = adj("T", DATES, c, c)[0]
    cut = int(np.searchsorted(DATES, np.datetime64("2019-01-31")))          # между первым и вторым событием
    _table(tmp_path, monkeypatch, EV[:1])                                  # второе событие префиксу неизвестно
    pref = adj("T", DATES[:cut], c[:cut], c[:cut])[0]
    return pref.tobytes() == full[:cut].tobytes()


def test_prefiks_pryamaya_pobaytno(tmp_path, monkeypatch):
    assert _prefiks_ok(dividendy.adjust_ohlc, tmp_path, monkeypatch)


def test_mutatsiya_korrektirovka_obratnaya_lovitsya(tmp_path, monkeypatch):
    """Падает до правки: обратная умножает прошлое на коэффициент будущей выплаты, префикс ≠ полный."""
    assert not _prefiks_ok(obratnaya, tmp_path, monkeypatch)


def test_dohodnosti_cherez_otsechku_kak_prezhde(tmp_path, monkeypatch):
    c = _price()
    _table(tmp_path, monkeypatch, EV)
    new, old = dividendy.adjust_ohlc("T", DATES, c, c)[0], obratnaya("T", DATES, c, c)[0]
    assert np.abs(np.diff(np.log(new)) - np.diff(np.log(old))).max() <= 1e-12
    g = int(np.searchsorted(DATES, np.datetime64("2018-06-15"), side="right"))
    assert new[g] / new[g - 1] == pytest.approx(c[g] / (c[g - 1] - 3.0), rel=1e-12)   # полная доходность с дивидендом
    assert new[:g].tobytes() == c[:g].tobytes()


def test_bez_sobytiy_ryad_syroy(tmp_path, monkeypatch):
    c = _price()
    _table(tmp_path, monkeypatch, [("X", "2018-06-15", 3.0)])
    assert dividendy.adjust_ohlc("T", DATES, c, c)[0].tobytes() == c.tobytes()
