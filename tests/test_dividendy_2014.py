"""Дивиденды 2014 в движке (PR dividendy-2014, Ж-113 п. 5): 6 выплат внутри В2, правило T+2 по барам, как 2015–2016.
Даты по ценовым гэпам — тем же критерием, что tests/test_dividendy_gap.py (окно ±4 бара от реестра, минимум |open/close(t−1) − 1 + D/c|).
Данные ≥ 2020 не читаются. python3 -m pytest tests/test_dividendy_2014.py -q -s"""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy

CUT = pd.Timestamp("2015-01-01")
WIN = 4


def _rows14():
    t = pd.read_csv(dividendy.TABLE_2014, parse_dates=["record_date"])
    return t[t["dps_rub"] > 0]


def _bars():
    from fastbt.daily_1850 import out_path
    if not out_path().exists():
        pytest.skip("нет ряда 18:50")
    d = pd.read_parquet(out_path(), columns=["ticker", "date", "open", "close"])
    d["date"] = pd.to_datetime(d["date"])
    return d[d["date"] < CUT].sort_values(["ticker", "date"])          # цены 2015+ не читаются


def _gap(b, rec, dps):
    """(день перед гэпом по ценам, |ошибка|, вторая ошибка, r в день гэпа, -D/c)."""
    dt, c, o = b["date"].values, b["close"].values, b["open"].values
    i = int(np.searchsorted(dt, np.datetime64(rec), side="right")) - 1
    cand = sorted((abs(o[t] / c[t - 1] - 1 + dps / c[t - 1]), t) for t in range(max(i - WIN, 1), min(i + WIN + 1, len(c))))
    t = cand[0][1]
    return pd.Timestamp(dt[t - 1]), cand[0][0], cand[1][0], o[t] / c[t - 1] - 1, -dps / c[t - 1]


def _engine(tic, rec, dps):
    e = dividendy.events(tic)
    rec = pd.Timestamp(rec)
    m = e[(e["dps_rub"] == dps) & (e["last_cum_date"] <= rec) & (e["last_cum_date"] > rec - pd.Timedelta(days=10))]
    return m["last_cum_date"].iloc[0] if len(m) == 1 else pd.NaT


def test_shest_vyplat_2014_chitayutsya_dvizhkom():
    r = _rows14()
    assert len(r) == 6
    assert (r["record_date"] >= "2014-08-26").all() and (r["record_date"] < "2015-01-01").all()
    for _, x in r.iterrows():
        assert pd.notna(_engine(x["ticker"], x["record_date"], x["dps_rub"])), (x["ticker"], x["record_date"])


def test_digest_vidit_tablicu_2014(tmp_path, monkeypatch):
    before = dividendy.digest()
    f = tmp_path / "d.csv"
    f.write_bytes(dividendy.TABLE_2014.read_bytes() + b"\n")
    monkeypatch.setattr(dividendy, "TABLE_2014", f)
    assert dividendy.digest() != before


def test_mutatsiya_bez_tablicy_2014_lovitsya(monkeypatch):
    """Без правки (таблица 2014 не в _tables) выплата PHOR 2014 не появляется в gap_divs — тест обязан падать."""
    monkeypatch.setattr(dividendy, "_tables", lambda: [dividendy.TABLE, dividendy.TABLE_2015])
    with pytest.raises(AssertionError):
        test_shest_vyplat_2014_chitayutsya_dvizhkom()


# Итог сверки, зафиксирован как есть (даты не подгонялись): гэп по ценам совпал с T+2 по барам у 4 из 6.
# PHOR 29.09 (D 1,9 %): по ценам минимум на 30.09 (ошибка 0,57 п.п., второй 0,79) — гэп не различим, движок 25.09.
# MGNT 10.10 (D 0,8 %): по ценам минимум 07.10 (ошибка 0,09 п.п., второй 0,13) — D ниже шума, движок 08.10.
MATCH = {("MTSS", "2014-10-14"), ("NVTK", "2014-10-27"), ("LKOH", "2014-12-26"), ("MGNT", "2014-12-30")}
NOGAP = {("PHOR", "2014-09-29"), ("MGNT", "2014-10-10")}


def _compare(shift=0):
    d = _bars()
    hit = set()
    for _, x in _rows14().iterrows():
        b = d[d["ticker"] == x["ticker"]].reset_index(drop=True)
        got = _engine(x["ticker"], x["record_date"], x["dps_rub"])
        j = int(np.searchsorted(b["date"].values, np.datetime64(got))) + shift
        if b["date"][j] == _gap(b, x["record_date"], x["dps_rub"])[0]:
            hit.add((x["ticker"], str(x["record_date"].date())))
    return hit


def test_gepy_2014_chetyre_iz_shesti_sovpadayut_dva_bez_gepa_perechisleny():
    hit = _compare()
    assert hit == MATCH, sorted(hit ^ MATCH)
    assert {(t, str(r.date())) for t, r in zip(_rows14()["ticker"], _rows14()["record_date"])} - hit == NOGAP


def test_mutatsii_sdvig_plyus_minus_den_lovyatsya():
    for k in (1, -1):
        assert _compare(shift=k) != MATCH, k
