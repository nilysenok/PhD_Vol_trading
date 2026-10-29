"""Лист R-в: дневной IMOEX на test1 из 10-минуток Z2 + index_d (#541-логика, общий модуль imoex_test1); train4 — прежний csv."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import golaya_s7_s9 as G  # noqa: E402
import imoex_test1 as IT  # noqa: E402
import r1_e  # noqa: E402
import r1_v as R  # noqa: E402


def _den(day, last_hhmm, base):
    ts = pd.date_range(f"{day} 10:00", f"{day} {last_hhmm}", freq="10min")
    return pd.DataFrame({"begin": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": [base - 0.5 + i for i in range(len(ts))],
                         "close": [base + i for i in range(len(ts))]})


@pytest.fixture
def faily(tmp_path, monkeypatch):
    for a in ("IMOEX", "START", "T0", "END"):
        monkeypatch.setattr(G, a, getattr(G, a))
    monkeypatch.setattr(R.AS, "PER", R.AS.PER)
    monkeypatch.setattr(R.AS, "GODY", R.AS.GODY)
    p10, pd_ = tmp_path / "m10.parquet", tmp_path / "d.parquet"
    pd.concat([_den("2021-12-29", "18:50", 100.0), _den("2021-12-30", "12:10", 5.0)]).to_parquet(p10)
    pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2024-12-30", "2025-01-09"]).date, "open": [9.0, 19.0, 700.0],
                  "close": [10.0, 20.0, 777.0]}).to_parquet(pd_)
    return p10, pd_


def test_test1_dnevnoj_imoex_bez_2025(faily):
    p10, pd_ = faily
    R.nastroit("test1", str(p10), str(pd_))
    im = G.chitat_imoex()
    assert list(im.columns) == ["OPEN", "CLOSE"] and im.index.max() < pd.Timestamp("2025-01-01")
    assert list(im.index) == [pd.Timestamp("2021-12-29"), pd.Timestamp("2022-01-03"), pd.Timestamp("2024-12-30")]   # обрыв 12-30 отброшен
    assert list(im["CLOSE"]) == [153.0, 10.0, 20.0]                 # 54 бара 10:00…18:50: close последнего = 100 + 53
    assert im["OPEN"].iloc[0] == 99.5 and list(im["OPEN"].iloc[1:]) == [9.0, 19.0]


def test_mutaciya_chtenie_bez_filtra_lovitsya(faily, monkeypatch):
    p10, pd_ = faily

    def bez_filtra(path, s, e, kol="close"):
        d = pd.read_parquet(path)
        return pd.Series(d[kol].to_numpy(float), index=pd.to_datetime(d["date"]))
    monkeypatch.setattr(IT, "dnevnye_index_d", bez_filtra)
    with pytest.raises(AssertionError):
        R.nastroit("test1", str(p10), str(pd_))
        assert G.chitat_imoex().index.max() < pd.Timestamp("2025-01-01")


def test_train4_csv_pobajtno_prezhnij(tmp_path, monkeypatch):
    csv = tmp_path / "IMOEX.csv"
    csv.write_text("BOARDID,TRADEDATE,CLOSE,OPEN\nSNDX,2016-01-04,1.5,1.0\nSNDX,2016-01-05,2.5,2.0\n", encoding="utf-8")
    monkeypatch.setattr(G, "IMOEX", csv)
    ref = pd.read_csv(csv, usecols=["TRADEDATE", "OPEN", "CLOSE"], parse_dates=["TRADEDATE"]).set_index("TRADEDATE")
    pd.testing.assert_frame_equal(G.chitat_imoex(), ref)


def test_r1_e_i_r1_v_odin_kod():
    assert r1_e.dnevka_iz_10min is IT.dnevka_iz_10min and r1_e.dnevnye_index_d is IT.dnevnye_index_d
