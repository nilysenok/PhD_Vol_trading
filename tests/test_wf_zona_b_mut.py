"""Тестировщик PR #739 (wf-L1b): мутации, не пойманные tests/test_wf_zona_b.py. Граница периода wf = 2026-03-01 исключительно:
строка ровно на границе (и на +1 день) обязана быть отрезана; train4/dev не уходит дальше 2021. Синтетика, без сети."""
import pandas as pd
import pytest

from fastbt import bars5m, data, istochniki_v6 as iv, period_v2, vselennaya_kachestvo as vk

DNI = ["2019-12-30", "2021-03-01", "2026-02-27", "2026-03-01", "2026-03-02"]


@pytest.fixture
def otmetka(tmp_path, monkeypatch):
    p = tmp_path / "wf_zamok_snyat.json"
    p.write_text("{}")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    return p


@pytest.fixture
def main(tmp_path):
    d = tmp_path / "m" / "moex_discovery" / "data" / "raw" / "candles_10m"
    d.mkdir(parents=True)
    ts = [pd.Timestamp(x) + pd.Timedelta(minutes=9 * 60 + 50 + 10 * i) for x in DNI for i in range(54)]
    for t in vk.TICKERS:
        pd.DataFrame({"timestamp": ts, "close": 1.0}).to_parquet(d / f"{t}.parquet")
    vk.default.cache_clear()
    yield tmp_path / "m"
    vk.default.cache_clear()


def test_kachestvo_granitsa_03_01_isklyuchitelno(main, otmetka):
    """Бары 2026-03-01 (ровно граница) и 03-02 не читаются; train4/None не видят ничего после 2019-12-30 (END=2020-01-01)."""
    assert vk.konec("wf") == pd.Timestamp("2026-03-01")
    assert vk.default(main, "wf").dates.max() == pd.Timestamp("2026-02-27")
    assert vk.konec(None) == vk.END == vk.konec("train4") == vk.konec("v2")
    assert vk.default(main).dates.max() == pd.Timestamp("2019-12-30")


def test_read_cut_granitsa_03_01_isklyuchitelno(tmp_path, otmetka):
    p = tmp_path / "v.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-02-27", "2026-03-01", "2026-03-02"]), "x": [1, 2, 3]}).to_parquet(p)
    assert list(data._read_cut(p, "wf")["x"]) == [1]


def test_bars5m_end_03_01_otkaz(tmp_path, otmetka):
    """Конец периода wf — 2026-02-28 включительно; end=2026-03-01 уже за концом."""
    src = tmp_path / "s"
    src.mkdir()
    for y in range(2022, 2027):
        pd.DataFrame({"begin": pd.to_datetime([f"{y}-02-02 10:00"]), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                      "volume": 1.0, "value": 1.0}).to_csv(src / f"AAA_{y}.csv", index=False)
    with pytest.raises(ValueError, match="позже конца"):
        bars5m.load_bars5m("AAA", end="2026-03-01", src=src, cache=tmp_path / "c")


@pytest.mark.parametrize("period", ["train4", "v2"])
def test_konec_dev_periodov_ne_dalshe_2021(period, otmetka):
    assert iv.konec(period) == pd.Timestamp("2022-01-01")
    assert iv.konec(period) < iv.konec("wf")
