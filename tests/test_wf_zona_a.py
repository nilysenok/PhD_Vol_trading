"""wf-L1a: граница чтения зависит от периода. wf (Ж-136): до 2026-02-28 и только при отметке снятого замка; прочие — прежняя.
Синтетика, данные не читаются. python3 -m pytest tests/test_wf_zona_a.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_sektor, period_v2, rezhim_e1, sobytiya_e1, zona

D = pd.Timestamp


@pytest.fixture(autouse=True)
def _dev():
    yield
    zona.set_zone("dev")


@pytest.fixture
def otmetka(tmp_path, monkeypatch):
    p = tmp_path / "wf_zamok_snyat.json"
    p.write_text("{}")
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    return p


@pytest.fixture
def bez_otmetki(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")


def _bars(path, days):
    rows = []
    for d in days:
        for k, t in enumerate(pd.date_range(f"{d} 10:00", f"{d} 18:40", freq="10min")):
            rows.append({"begin": t.strftime("%Y-%m-%d %H:%M:%S"), "open": 100.0 + k, "high": 101.0 + k,
                         "low": 99.0 + k, "close": 100.0 + k + 0.1 * (k % 3), "value": 1.0, "volume": 1.0})
    pd.DataFrame(rows).to_parquet(path)


def _wf_bez_otmetki_otkaz() -> bool:
    try:
        zona.set_zone("wf")
    except period_v2.ZamokVladeltsa:
        return zona.name() == "dev"
    return False


def _wf_chitaet_2026() -> bool:
    zona.set_zone("wf")
    zona.check(D("2026-02-28"), "t")
    zona.check(D("2025-06-02"), "t")
    return zona.end() == D("2026-03-01")


def _wf_zakryt_s_marta() -> bool:
    zona.set_zone("wf")
    try:
        zona.check(D("2026-03-01"), "t")
    except PermissionError:
        return True
    return False


def test_wf_bez_otmetki_otkaz(bez_otmetki):
    assert _wf_bez_otmetki_otkaz()
    assert zona.end() == D("2022-01-01")


def test_wf_s_otmetkoj_chitaet_do_2026_02_28_i_ne_dalshe(otmetka):
    assert _wf_chitaet_2026()
    assert _wf_zakryt_s_marta()
    assert zona.hard() == zona.END_WF


def test_prochie_zony_pobajtno_prezhnie(otmetka):
    assert zona.HARD == D("2025-01-01") and zona.END_DEV == D("2022-01-01") and zona.END_Z2 == D("2025-01-01")
    for z, e in (("dev", "2022-01-01"), ("test1", "2025-01-01")):
        with zona.zone_ctx(z):
            assert zona.end() == D(e) and zona.hard() == zona.HARD
            zona.check(D("2024-12-31"), "t")
            with pytest.raises(PermissionError, match="2025-01-01"):
                zona.check(D("2025-01-01"), "t")
    with pytest.raises(FileNotFoundError):
        with zona.zone_ctx("test1"):
            zona.path("daily", None)
    assert zona.path("daily", "x") == "x"
    with zona.zone_ctx("wf"):
        assert zona.path("daily", "x") == "x"      # файлы wf — по путям dev
    assert zona.name() == "dev"


def test_sobytiya_wf_rezhet_po_2026_02_28(otmetka):
    dates = pd.bdate_range("2026-02-20", "2026-02-27")
    with pytest.raises(PermissionError, match="2022-01-01"):
        sobytiya_e1.cbr_block(dates, pd.DataFrame({"date": [], "known_from": []}))
    with zona.zone_ctx("wf"):
        assert len(sobytiya_e1.cbr_block(dates, pd.DataFrame({"date": [], "known_from": []}))) == len(dates)
        with pytest.raises(PermissionError, match="2026-03-01"):
            sobytiya_e1.cbr_block(pd.bdate_range("2026-02-25", "2026-03-03"), pd.DataFrame({"date": [], "known_from": []}))


def test_imoex_daily_wf_do_2026_02_27_dev_do_2021(tmp_path, otmetka):
    p = tmp_path / "IMOEX.parquet"
    _bars(p, ["2021-12-29", "2021-12-30", "2022-01-11", "2026-02-27", "2026-03-02"])
    dev = rezhim_e1.imoex_daily(p)
    assert dev["date"].max() == D("2021-12-30") and len(dev) == 2
    with zona.zone_ctx("wf"):
        wf = rezhim_e1.imoex_daily(p)
    assert wf["date"].max() == D("2026-02-27") and len(wf) == 4
    assert wf["date"].iloc[:2].tolist() == dev["date"].tolist()
    np.testing.assert_array_equal(wf["close"].iloc[:2].to_numpy(), dev["close"].to_numpy())


def test_load_indexes_dev_pobajtno_wf_do_2026(tmp_path, otmetka):
    _bars(tmp_path / "MOEXOG.parquet", ["2021-12-29", "2021-12-30", "2022-01-11", "2026-02-27", "2026-03-02"])
    dev = e1_sektor.load_indexes(["MOEXOG"], main=tmp_path)["MOEXOG"]
    raw = pd.read_parquet(tmp_path / "MOEXOG.parquet")
    raw = raw[raw["begin"] < "2022-01-01"].copy()
    raw["timestamp"] = pd.to_datetime(raw["begin"])
    pd.testing.assert_series_equal(dev, e1_sektor.idx_daily(raw, "MOEXOG"))
    assert dev.index.max() == D("2021-12-30")
    with pytest.raises(PermissionError):
        e1_sektor.load_indexes(["MOEXOG"], main=tmp_path, end=D("2022-06-01"))
    with zona.zone_ctx("wf"):
        wf = e1_sektor.load_indexes(["MOEXOG"], main=tmp_path)["MOEXOG"]
        assert wf.index.max() == D("2026-02-27") and len(wf) == 4
        with pytest.raises(PermissionError):
            e1_sektor.load_indexes(["MOEXOG"], main=tmp_path, end=D("2026-04-01"))
    pd.testing.assert_series_equal(wf.iloc[:2], dev)


def test_mutaciya_granica_2026_bez_proverki_otmetki_lovitsya(bez_otmetki, monkeypatch):
    assert _wf_bez_otmetki_otkaz()
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    assert not _wf_bez_otmetki_otkaz()      # мутация: wf открыт без отметки — тест выше упал бы


def test_mutaciya_granica_2025_ostalas_lovitsya(otmetka, monkeypatch):
    assert _wf_chitaet_2026()
    zona.set_zone("dev")
    monkeypatch.setattr(zona, "hard", lambda: zona.HARD)
    with pytest.raises(PermissionError):
        _wf_chitaet_2026()                   # мутация: HARD = 2025 в wf — чтение 2026-02-28 отказано


def test_mutaciya_wf_bez_verhnej_granicy_lovitsya(otmetka, monkeypatch):
    assert _wf_zakryt_s_marta()
    zona.set_zone("dev")
    monkeypatch.setattr(zona, "hard", lambda: D("2099-01-01"))
    assert not _wf_zakryt_s_marta()
