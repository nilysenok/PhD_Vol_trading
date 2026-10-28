"""Дивиденды 2015–2016 в движке (PR данных К-в1): читаются обе таблицы, digest (Д5) берёт обе, дни 2017–2021 прежние.
Календарь — синтетический (пн–пт), данные баров не читаются; тест правила T+2 на реальном календаре — при наличии ряда 18:50.
python3 -m pytest tests/test_dividendy_2015.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy
from fastbt.data import TICKERS

CAL = pd.bdate_range("2014-06-01", "2021-12-31").values.astype("datetime64[ns]")
DATES16 = pd.bdate_range("2016-01-04", "2016-12-30").values
DATES17_21 = pd.bdate_range("2017-01-02", "2021-12-30").values


@pytest.fixture(autouse=True)
def _cal(monkeypatch):
    monkeypatch.setattr(dividendy, "_calendar", lambda: CAL)


def _check_2016_otsechka():
    """ALRS: реестр 19.07.2016 (вт) → last_cum 15.07 (пт), день гэпа 18.07 — D = 2,09."""
    d = dividendy.gap_divs("ALRS", DATES16)
    g = int(np.searchsorted(DATES16, np.datetime64("2016-07-18")))
    assert d[g] == pytest.approx(2.09)
    assert d.sum() == pytest.approx(2.09)                      # в 2016 у ALRS одна выплата


def _check_digest_vidit_tablicu_2015(tmp_path, monkeypatch):
    before = dividendy.digest()
    f = tmp_path / "d.csv"
    f.write_bytes(dividendy.TABLE_2015.read_bytes() + b"\n")
    monkeypatch.setattr(dividendy, "TABLE_2015", f)
    assert dividendy.digest() != before


def test_otsechka_2016_daet_D_v_gap_divs():
    _check_2016_otsechka()


def test_mutatsiya_chitaetsya_tolko_tablitsa_2017_2021_lovitsya(monkeypatch):
    monkeypatch.setattr(dividendy, "_tables", lambda: [dividendy.TABLE])
    with pytest.raises(AssertionError):
        _check_2016_otsechka()


def test_digest_menyaetsya_pri_podmene_tablicy_2015(tmp_path, monkeypatch):
    _check_digest_vidit_tablicu_2015(tmp_path, monkeypatch)


def test_mutatsiya_digest_bez_tablicy_2015_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(dividendy, "_tables", lambda: [dividendy.TABLE])
    with pytest.raises(AssertionError):
        _check_digest_vidit_tablicu_2015(tmp_path, monkeypatch)


def test_digest_menyaetsya_pri_podmene_tablicy_2017(tmp_path, monkeypatch):
    before = dividendy.digest()
    f = tmp_path / "d.csv"
    f.write_bytes(dividendy.TABLE.read_bytes() + b"\n")
    monkeypatch.setattr(dividendy, "TABLE", f)
    assert dividendy.digest() != before


def test_stroki_bez_vyplat_propuskayutsya():
    for t in ("AFLT", "MTLR"):
        assert (dividendy.events(t)["last_cum_date"] >= pd.Timestamp("2017-01-01")).all()
        assert (dividendy.gap_divs(t, DATES16) == 0).all()


def test_golden_dni_2017_2021_pobaytno_prezhnie():
    old = pd.read_csv(dividendy.TABLE, parse_dates=["last_cum_date"])
    for t in TICKERS:
        o = old[(old["ticker"] == t) & (old["dps_rub"] > 0)].sort_values("last_cum_date")
        want = np.zeros(len(DATES17_21))
        for cum, dps in zip(o["last_cum_date"], o["dps_rub"]):
            g = int(np.searchsorted(DATES17_21, np.datetime64(cum), side="right"))
            if 0 < g < len(DATES17_21):
                want[g] += float(dps)
        assert dividendy.gap_divs(t, DATES17_21).tobytes() == want.tobytes(), t


def test_last_cum_t_plus_2_s_vyhodnymi():
    r = dividendy.last_cum(pd.Series(pd.to_datetime(["2016-07-19", "2016-06-11", "2015-01-11"])))
    assert list(r) == list(pd.to_datetime(["2016-07-15", "2016-06-08", "2015-01-07"]))


def test_last_cum_do_nachala_kalendarya_oshibka():
    with pytest.raises(ValueError):
        dividendy.last_cum(pd.Series(pd.to_datetime(["2014-06-02"])))


def test_pravilo_t2_na_realnom_kalendare_sovpadaet_s_tablicey_2017(monkeypatch):
    monkeypatch.undo()
    from fastbt.daily_1850 import out_path
    if not out_path().exists():
        pytest.skip("нет ряда 18:50")
    old = pd.read_csv(dividendy.TABLE, parse_dates=["record_date", "last_cum_date"])
    old = old[old["dps_rub"] > 0]
    ok = (dividendy.last_cum(old["record_date"]) == old["last_cum_date"]).mean()
    assert ok >= 0.9, ok
