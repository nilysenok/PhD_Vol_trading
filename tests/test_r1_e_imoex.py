"""Лист R-е: дневка IMOEX из 10-минуток (parquet Z2) — правило «close последнего бара полного дня»."""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("r1_e", ROOT / "scripts/revers/r1_e.py")
r1e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1e)
from fastbt import daily_1850 as d18  # noqa: E402

S, E = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")


def _den(day, last_hhmm, base):
    ts = pd.date_range(f"{day} 10:00", f"{day} {last_hhmm}", freq="10min")
    return pd.DataFrame({"begin": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": [base - 0.5 + i for i in range(len(ts))], "close": [base + i for i in range(len(ts))]})


def _bars():
    """3 дня: полный до 18:50 (close последнего бара 18:50), полный до 18:40, дыра до 12:10."""
    return pd.concat([_den("2024-03-04", "18:50", 100.0), _den("2024-03-05", "18:40", 200.0), _den("2024-03-07", "12:10", 300.0)])


def _proverka(fn):
    r = fn(_bars(), S, E)
    n1, n2 = len(pd.date_range("2024-03-04 10:00", "2024-03-04 18:50", freq="10min")), len(pd.date_range("2024-03-05 10:00", "2024-03-05 18:40", freq="10min"))
    assert list(r.index) == [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")]      # дыра 03-07 не взята
    assert list(r.values) == [100.0 + n1 - 1, 200.0 + n2 - 1]                                 # close последнего бара


def test_pravilo_dnevki_izvestnyj_otvet():
    _proverka(r1e.dnevka_iz_10min)


def test_mutaciya_close_1850_vmesto_poslednego_bara_lovitsya():
    def mut(bars, s, e):
        b = bars.assign(timestamp=pd.to_datetime(bars["begin"]), open=1.0, high=1.0, low=1.0, volume=1)
        return d18.aggregate(b, "IMOEX").set_index("date")["close"]
    with pytest.raises(AssertionError):
        _proverka(mut)


def test_mutaciya_dyra_ne_otbrasyvaetsya_lovitsya():
    def mut(bars, s, e):
        b = bars.assign(t=pd.to_datetime(bars["begin"]))
        return b.groupby(b.t.dt.normalize())["close"].last()
    with pytest.raises(AssertionError):
        _proverka(mut)


def test_otkaz_data_2025_i_pozzhe():
    bad = pd.concat([_bars(), _den("2025-01-09", "18:50", 1.0)])
    with pytest.raises(PermissionError):
        r1e.dnevka_iz_10min(bad, S, E)
    with pytest.raises(PermissionError):
        r1e.dnevka_iz_10min(_bars(), S, pd.Timestamp("2025-02-01"))


def test_parquet_idet_tem_zhe_kodom(tmp_path):
    p = tmp_path / "IMOEX.parquet"
    _bars().to_parquet(p)
    pd.testing.assert_series_equal(r1e.zagruzit_imoex(p, S, E), r1e.dnevka_iz_10min(_bars(), S, E))


def test_sverka_2019_s_iss_csv():
    main = Path(os.environ.get("FASTBT_MAIN", ROOT))
    csv, pq_ = main / "data/iss_mix_2015_2019/IMOEX.csv", main / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet"
    if not (csv.exists() and pq_.exists()):
        pytest.skip("нет IMOEX 2015–2019 / 10-минуток в этой копии")
    s, e = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")
    a, c = r1e.zagruzit_imoex(pq_, s, e), r1e.zagruzit_imoex(csv, s, e)
    j = pd.concat([a.rename("a"), c.rename("c")], axis=1)
    assert j.a.notna().sum() >= 225 and j.c.notna().sum() == 252
    assert ((j.a / j.c - 1).abs().dropna() < 1e-9).all()                                       # допуск: побайтно


def _index_d(tmp_path):
    d = pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2024-12-30", "2025-01-09", "2026-03-02"]).date, "open": [9.0, 19.0, 700.0, 800.0], "close": [10.0, 20.0, 777.0, 888.0]})
    p = tmp_path / "IMOEX_d.parquet"
    d.to_parquet(p)
    return p


def _proverka_d(fn, p):
    r = fn(p, pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"))
    assert r.index.max() < pd.Timestamp("2025-01-01") and list(r.values) == [10.0, 20.0]


def test_index_d_ne_chitaet_2025(tmp_path):
    p = _index_d(tmp_path)
    _proverka_d(r1e.dnevnye_index_d, p)
    with pytest.raises(PermissionError):
        r1e.dnevnye_index_d(p, pd.Timestamp("2022-01-01"), pd.Timestamp("2025-06-01"))


def test_mutaciya_chtenie_bez_filtra_lovitsya(tmp_path):
    import pyarrow.parquet as pq

    def mut(p, s, e):
        d = pq.read_table(p).to_pandas()
        return pd.Series(d["close"].to_numpy(), index=pd.to_datetime(d["date"]))
    with pytest.raises(AssertionError):
        _proverka_d(mut, _index_d(tmp_path))


def test_stik_10min_do_2022_index_d_posle(tmp_path):
    bars = pd.concat([_den("2021-12-30", "18:50", 100.0), _den("2021-12-31", "12:10", 5.0)])
    p10 = tmp_path / "m10.parquet"
    bars.to_parquet(p10)
    r = r1e.zagruzit_imoex(p10, pd.Timestamp("2021-01-01"), E, _index_d(tmp_path))
    assert list(r.index) == [pd.Timestamp("2021-12-30"), pd.Timestamp("2022-01-03"), pd.Timestamp("2024-12-30")]
    assert r.iloc[-1] == 20.0 and r.index.is_monotonic_increasing
