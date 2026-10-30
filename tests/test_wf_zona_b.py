"""Лист wf-L1b: границы чтения периода wf (06.2014–2026-02-28) в bars5m, vselennaya, data._read_cut, istochniki_v6.konec,
vselennaya_kachestvo / period_v2.valid_vec и ставке ЦБ. Синтетика, без сети; отметка замка подменяется через period_v2.OTMETKA."""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars5m, data, istochniki_v6 as iv, msp, period_v2, vselennaya as V, vselennaya_kachestvo as vk

DNI_54 = ["2019-12-30", "2021-03-01", "2025-03-03", "2026-02-27", "2026-03-02"]


@pytest.fixture
def otmetka(tmp_path, monkeypatch):
    p = tmp_path / "wf_zamok_snyat.json"
    p.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64 + '", "data_snyatiya": "2026-10-30"}')
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    return p


@pytest.fixture
def bez_otmetki(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")


@pytest.fixture
def main(tmp_path):
    """Десятиминутки 17 тикеров: по 54 бара окна [09:50; 18:50) в дни DNI_54."""
    d = tmp_path / "m" / "moex_discovery" / "data" / "raw" / "candles_10m"
    d.mkdir(parents=True)
    ts = [pd.Timestamp(x) + pd.Timedelta(minutes=9 * 60 + 50 + 10 * i) for x in DNI_54 for i in range(54)]
    for t in vk.TICKERS:
        pd.DataFrame({"timestamp": ts, "close": 1.0}).to_parquet(d / f"{t}.parquet")
    vk.default.cache_clear()
    yield tmp_path / "m"
    vk.default.cache_clear()


# --- vselennaya_kachestvo / valid_vec ---------------------------------------------------------------------------------------
def _den_2020_plus_valen(main, vec=period_v2.valid_vec):
    ok = vec("SBER", pd.to_datetime(["2021-03-01", "2025-03-03", "2026-02-27"]), period="wf", main=main)
    assert ok.all(), ok       # 54 бара — день ≥ 2020 валиден


def test_den_posle_2020_s_54_barami_validen(main, otmetka):
    _den_2020_plus_valen(main)


def test_wf_bez_otmetki_otkaz_kachestvo(main, bez_otmetki):
    with pytest.raises(period_v2.ZamokVladeltsa):
        vk.default(main, "wf")


def test_train4_pobaytno_prezhnee(main, otmetka):
    a, b, c = vk.default(main), vk.default(main, "train4"), vk.default(main, "v2")
    pd.testing.assert_frame_equal(a.counts, b.counts)
    pd.testing.assert_frame_equal(a.counts, c.counts)
    assert a.dates.max() == pd.Timestamp("2019-12-30") and vk.konec("train4") == vk.END
    np.testing.assert_array_equal(period_v2.valid_vec("SBER", pd.to_datetime(["2019-12-30", "2021-03-01"]), main=None, kach=a), [True, False])


def test_wf_bary_posle_konca_ne_chitayutsya(main, otmetka):
    assert vk.default(main, "wf").dates.max() == pd.Timestamp("2026-02-27")      # 2026-03-02 отрезан


def test_data_za_kontsom_kalendarya_otkaz(main, otmetka):
    with pytest.raises(ValueError, match="календарь баров кончается"):
        period_v2.valid_vec("SBER", pd.to_datetime(["2026-02-28", "2026-03-20"]), period="wf", main=main)
    period_v2.valid_vec("SBER", pd.to_datetime(["2026-02-28"]), period="wf", main=main)      # выходные сразу за концом — ok


def test_mutaciya_end_2020_dlya_wf_lovitsya(main, otmetka, monkeypatch):
    monkeypatch.setattr(vk, "konec", lambda period=None: vk.END)
    with pytest.raises((AssertionError, ValueError)):
        _den_2020_plus_valen(main)


def test_mutaciya_fill_value_false_lovitsya(main, otmetka):
    def staryj(ticker, dates, kach=None, period=None, main=None):       # прежняя valid_vec: молча False за календарём
        kach = kach or vk.default(main, period)
        return kach.valid[ticker].reindex(pd.DatetimeIndex(dates), fill_value=False).to_numpy(bool)
    def za_kalendarem_otkaz(vec):
        with pytest.raises(ValueError):
            vec("SBER", pd.to_datetime(["2026-03-20"]), period="wf", main=main)
    za_kalendarem_otkaz(period_v2.valid_vec)
    with pytest.raises(pytest.fail.Exception):       # прежняя функция молча вернула False вместо отказа
        za_kalendarem_otkaz(staryj)


# --- ставка ЦБ ------------------------------------------------------------------------------------------------------------------
def _stavka_otkazyvaet(fn):
    konec = msp.rate_cbr().index[-1]                  # фактический конец ряда, не зашитая дата
    with pytest.raises(ValueError):
        fn(pd.date_range(konec - pd.Timedelta(days=2), konec + pd.Timedelta(days=1)))


def test_stavka_za_kontsom_ryada_otkaz():
    _stavka_otkazyvaet(msp.rate_on)
    r = msp.rate_cbr()
    assert msp.rate_on(r.index[-3:]).notna().all()                    # внутри ряда — как было
    assert msp.rate_on(pd.date_range("2021-12-01", "2021-12-31")).notna().all()


def test_mutaciya_ffill_stavki_lovitsya():
    def staryj(days):                                              # прежний rate_on: ffill вперёд за конец ряда
        r, days = msp.rate_cbr(), pd.DatetimeIndex(days)
        return r.reindex(r.index.union(days)).sort_index().ffill().reindex(days)
    with pytest.raises(pytest.fail.Exception):
        _stavka_otkazyvaet(staryj)


# --- граница wf только при отметке ----------------------------------------------------------------------------------------------
def _wf_bez_otmetki_otkaz():
    with pytest.raises(PermissionError):
        iv.konec("wf")
    with pytest.raises(ValueError):
        V.sostav("2022-01-03")


def test_vselennaya_i_konec_bez_otmetki_otkaz(bez_otmetki):
    _wf_bez_otmetki_otkaz()


def test_mutaciya_granitsa_bez_otmetki_lovitsya(bez_otmetki, monkeypatch):
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    with pytest.raises(pytest.fail.Exception):
        _wf_bez_otmetki_otkaz()


def test_konec_wf_s_otmetkoy(otmetka):
    assert iv.konec("wf") == pd.Timestamp("2026-03-01")
    assert iv.konec("train4") == pd.Timestamp("2022-01-01")


def test_vselennaya_wf_sostav_i_oborot(otmetka, tmp_path):
    assert len(V.sostav("2022-01-10")) > 0                           # 2021-12-17 + 120 дней
    with pytest.raises(ValueError, match="Д4"):
        V.sostav("2022-06-01")                                       # состав 2022+ не подключён — молча не тянется
    with pytest.raises(ValueError, match="позже конца"):
        V.sostav("2027-01-05")
    (tmp_path / "daily_value").mkdir()
    pd.DataFrame({"date": ["2021-12-30"], "value": [5.0]}).to_csv(tmp_path / "daily_value" / "A.csv", index=False)
    assert V.oborot_goda("A", 2021, str(tmp_path)) == 5.0
    with pytest.raises(ValueError, match="нет строк года"):
        V.oborot_goda("A", 2022, str(tmp_path))
    assert V.oborot_goda("B", 2019, str(tmp_path)) == 0.0            # ≤ 2021: прежнее поведение


# --- bars5m ---------------------------------------------------------------------------------------------------------------------
def _csv(src, t, y, day):
    src.mkdir(exist_ok=True)
    pd.DataFrame({"begin": pd.to_datetime([f"{day} 10:00"]), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                  "volume": 1.0, "value": 1.0}).to_csv(src / f"{t}_{y}.csv", index=False)


def test_bars5m_wf(tmp_path, bez_otmetki, monkeypatch):
    src = tmp_path / "s"
    for y, day in [(2021, "2021-12-30"), (2022, "2022-02-01"), (2025, "2025-02-03"), (2026, "2026-02-27")]:
        _csv(src, "AAA", y, day)
    kw = dict(src=src, cache=tmp_path / "c")
    with pytest.raises(period_v2.ZamokVladeltsa):
        bars5m.load_bars5m("AAA", end="2026-02-28", **kw)
    out = bars5m.load_bars5m("AAA", **kw)                            # умолчание до 2021 — отметка не нужна
    assert out.date.max() == pd.Timestamp("2021-12-30")
    p = tmp_path / "otm.json"
    p.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64 + '", "data_snyatiya": "2026-10-30"}')
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    with pytest.raises(FileNotFoundError, match="2023"):
        bars5m.load_bars5m("AAA", end="2026-02-28", **kw)            # нет файла 2023 — не молчаливый пустой год
    for y in (2023, 2024):
        _csv(src, "AAA", y, f"{y}-02-01")
    full = bars5m.load_bars5m("AAA", end="2026-02-28", **kw)
    assert full.date.max() == pd.Timestamp("2026-02-27")
    with pytest.raises(ValueError, match="позже конца"):
        bars5m.load_bars5m("AAA", end="2026-03-31", **kw)
    pd.testing.assert_frame_equal(bars5m.load_bars5m("AAA", **kw), out)       # train4/dev — побайтно прежнее


# --- data._read_cut ---------------------------------------------------------------------------------------------------------------
def test_read_cut_wf(tmp_path, bez_otmetki, monkeypatch):
    p = tmp_path / "v.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2021-12-30", "2025-06-02", "2026-02-27", "2026-03-02"]), "x": [1, 2, 3, 4]}).to_parquet(p)
    assert len(data._read_cut(p)) == 4                                # dev — как было
    with pytest.raises(period_v2.ZamokVladeltsa):
        data._read_cut(p, "wf")
    o = tmp_path / "o.json"
    o.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "' + "a" * 64 + '", "data_snyatiya": "2026-10-30"}')
    monkeypatch.setattr(period_v2, "OTMETKA", o)
    assert list(data._read_cut(p, "wf")["x"]) == [1, 2, 3]            # 2025 не режется HARD-ом, 2026-03-02 отрезан
