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
    return pd.DataFrame({"begin": ts.strftime("%Y-%m-%d %H:%M:%S"), "close": [base + i for i in range(len(ts))]})


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
