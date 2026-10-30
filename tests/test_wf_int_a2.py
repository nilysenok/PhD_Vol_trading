"""wf-int-a2: период wf включает зону wf; календарь дивидендов, состав IMOEX и сектора 2022–2026 в зоне wf; ≤ 2021 побайтно прежнее.

Синтетика (календарь, свод, состав) и реальные таблицы состава в docs/. Данные рядов и сеть не нужны.
Мутации: «календарь ≤ 2021 молча в wf», «состав виден до known_from», «сектора 2021 тянутся вперёд»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850, dividendy as dv, e1_rel_sektor as rs, e1_sektor as sek, period_v2 as pv, s9, vselennaya as vs, zona

DOCS = vs.REPO / "docs" / "issledovaniya"


def otkaz(exc, fn, match=""):
    """Отказ exc обязателен; его отсутствие — AssertionError (его ловит тест мутации, pytest.fail он не ловит)."""
    try:
        fn()
    except exc as e:
        assert match in str(e), str(e)
        return
    raise AssertionError(f"не было отказа {exc.__name__}")


@pytest.fixture(autouse=True)
def _chistka():
    dv._calendar.cache_clear()
    yield
    zona.set_zone("dev")
    s9.set_period("train")
    dv._calendar.cache_clear()


# ---------- период → зона ----------
def test_train4_zona_dev_i_prezhnie_gody():
    s9.set_period("train4")
    assert zona.name() == "dev" and s9.YEARS == (2016, 2017, 2018, 2019) and s9.VAL_START == 2015


def test_wf_vklyuchaet_zonu_i_vozvrashchaet():
    with s9.period_ctx("wf"):
        assert zona.name() == "wf" and s9.PERIOD == "wf" and s9.YEARS[0] == 2015 and s9.YEARS[-1] == 2026
        assert s9.VAL_START == 2014
    assert zona.name() == "dev" and s9.PERIOD == "train"


def test_wf_bez_otmetki_otkaz_period_ne_menyaetsya(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "OTMETKA", tmp_path / "net.json")
    with pytest.raises(pv.ZamokVladeltsa):
        s9.set_period("wf")
    assert s9.PERIOD == "train" and zona.name() == "dev"


# ---------- календарь дивидендов ----------
def _parquet(tmp_path, monkeypatch, do):
    d = pd.bdate_range("2014-06-02", do)
    p = tmp_path / "daily.parquet"
    pd.DataFrame({"date": d}).to_parquet(p)
    monkeypatch.setattr(daily_1850, "out_path", lambda main=None: p)
    return d.values.astype("datetime64[ns]")


def inv_kalendar_wf_ne_tihoe_2021(tmp_path, monkeypatch):
    _parquet(tmp_path, monkeypatch, "2021-12-30")
    with zona.zone_ctx("wf"):
        dv._calendar.cache_clear()
        otkaz(FileNotFoundError, dv._calendar, "ряд wf")


def test_kalendar_wf_korotkij_otkaz(tmp_path, monkeypatch):
    inv_kalendar_wf_ne_tihoe_2021(tmp_path, monkeypatch)


def test_kalendar_dev_korotkij_prezhnij(tmp_path, monkeypatch):
    cal = _parquet(tmp_path, monkeypatch, "2021-12-30")
    assert (dv._calendar() == cal).all()


def test_mutaciya_kalendar_2021_v_wf_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(dv, "WF_KALENDAR_DO", "2000-01-01")           # проверка снята — календарь ≤ 2021 проходит молча
    with pytest.raises(AssertionError):
        inv_kalendar_wf_ne_tihoe_2021(tmp_path, monkeypatch)


def test_dividendy_2023_iz_svoda_korrektiruyut_ryad_v_wf(tmp_path, monkeypatch):
    cal = _parquet(tmp_path, monkeypatch, "2026-02-27")
    shapka = "ticker,record_date,dps_rub,announce_pit,godna,godna_korr,prichina\n"
    svod, pustoj = tmp_path / "svod.csv", tmp_path / "pustoj.csv"
    svod.write_text(shapka + "SBER,2023-08-10,20.0,,0,1,x\n")
    pustoj.write_text(shapka)
    monkeypatch.delenv("FASTBT_Z2_DIV", raising=False)
    close = np.full(len(cal), 100.0)
    g = int(np.searchsorted(cal, np.datetime64("2023-08-10")))      # T+1: last_cum 09.08, день гэпа 10.08
    with zona.zone_ctx("wf"):
        monkeypatch.setattr(dv, "SVOD", pustoj)
        (base,) = dv.adjust_ohlc("SBER", cal, close, close)
        monkeypatch.setattr(dv, "SVOD", svod)
        assert dv.gap_divs("SBER", cal)[g] == 20.0
        (adj,) = dv.adjust_ohlc("SBER", cal, close, close)
    r = adj / base                                                  # прежние таблицы ≤ 2021 одинаковы в обоих прогонах
    assert (r[:g] == 1.0).all() and r[g:] == pytest.approx(1 / 0.8)


# ---------- состав IMOEX ----------
def _new_row(de: str) -> frozenset:
    t = pd.read_csv(vs.SOSTAV_2022)
    return frozenset(t.loc[t["date_eff"] == de, "tickers"].iloc[0].split())


def test_sostav_2024_06_iz_novoj_tablicy():
    with zona.zone_ctx("wf"):
        assert vs.sostav("2024-06-25") == _new_row("2024-06-21")
        assert vs.sostav("2024-06-20") == _new_row("2024-03-22")      # known_from 30.05 < date_eff 21.06: до date_eff — прежний
        assert vs.sostav("2024-06-21") == _new_row("2024-06-21")


def test_sostav_do_2021_pobajtno_prezhnij_v_wf_i_dev():
    old = vs._sostav_table(str(vs.SOSTAV), vs.SOSTAV_COL)
    for d in ("2016-03-01", "2019-12-31", "2021-12-30"):
        want = [tk for de, tk in old if de <= pd.Timestamp(d)][-1]
        assert vs.sostav(d) == want
        with zona.zone_ctx("wf"):
            assert vs.sostav(d) == want


def test_sostav_wf_ne_menyaet_hesh_dev():
    h = vs.digest(root="/nonexistent")
    with zona.zone_ctx("wf"):
        assert vs.digest(root="/nonexistent") != h
    assert vs.digest(root="/nonexistent") == h


def _tablicy(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.write_text("date_eff,known_from,tickers\n2021-12-17,2021-12-03,AAA BBB\n")
    new.write_text("date_eff,known_from,tickers\n2021-12-17,2021-12-03,AAA BBB\n2022-01-10,2022-01-20,AAA CCC\n")
    return old, new


def inv_sostav_ne_vidno_do_known_from(old, new):
    with zona.zone_ctx("wf"):
        assert vs.sostav("2022-01-15", old) == frozenset({"AAA", "BBB"})      # date_eff прошёл, known_from — нет
        assert vs.sostav("2022-01-20", old) == frozenset({"AAA", "CCC"})


def test_sostav_ne_vidno_do_known_from(tmp_path, monkeypatch):
    old, new = _tablicy(tmp_path)
    monkeypatch.setattr(vs, "SOSTAV", old)
    monkeypatch.setattr(vs, "SOSTAV_2022", new)
    inv_sostav_ne_vidno_do_known_from(old, new)


def test_mutaciya_sostav_vidno_do_known_from_lovitsya(tmp_path, monkeypatch):
    old, new = _tablicy(tmp_path)
    monkeypatch.setattr(vs, "SOSTAV", old)
    monkeypatch.setattr(vs, "SOSTAV_2022", new)
    monkeypatch.setattr(vs, "_vidno_s", lambda de, kf: de)
    with pytest.raises(AssertionError):
        inv_sostav_ne_vidno_do_known_from(old, new)


# ---------- сектора ----------
def test_sektor_2023_iz_novoj_tablicy_v_wf_a_v_dev_net():
    dev = rs.sektor_pit()
    assert max(d for rows in dev.values() for d, _ in rows) <= pd.Timestamp("2021-12-31")
    with zona.zone_ctx("wf"):
        pit = rs.sektor_pit()
    assert pit["SBER"][-1][0] == pd.Timestamp("2025-12-22")
    dt = pd.DatetimeIndex(["2023-07-03"])
    assert list(rs._names(pit["SBER"], dt)) == ["MOEXFN"] and list(rs._names(pit["LSRG"], dt)) == ["MOEXRE"]
    assert any(d.year == 2023 for d, _ in pit["SBER"])
    for t, rows in dev.items():                         # ≤ 2021 те же строки
        assert pit[t][:len(rows)] == rows


def _close(d0="2023-01-02", d1="2023-03-31"):
    c = pd.Series(100.0, index=pd.bdate_range(d0, d1))
    return c, pd.Series(0.01, index=c.index)


def inv_sektor_2021_ne_perenositsya():
    c, sg = _close()
    ctx = rs.Ctx({}, {"SBER": c}, {"SBER": sg}, rs.sektor_pit())          # зона dev: строки только ≤ 2021
    otkaz(ValueError, lambda: ctx.rel("SBER", 5), "не переносится")


def test_sektor_2021_vpered_otkaz():
    inv_sektor_2021_ne_perenositsya()


def test_mutaciya_sektora_2021_vpered_lovitsya(monkeypatch):
    monkeypatch.setattr(sek, "PIT_STALE_DAYS", 10 ** 6)
    with pytest.raises(AssertionError):
        inv_sektor_2021_ne_perenositsya()


def test_sektor_wf_2023_ctx_ne_otkazyvaet():
    c, sg = _close()
    with zona.zone_ctx("wf"):
        _, names = rs.Ctx({}, {"SBER": c}, {"SBER": sg}, rs.sektor_pit()).rel("SBER", 5)
    assert set(names) == {"MOEXFN"}


def test_staticheskaya_karta_wf_podtverzhdena_i_otkaz(tmp_path):
    real = vs.REPO / "docs/issledovaniya/2026-10-27_sektory.csv"
    sek.proverit_sektory_wf(real)                                         # dev: ничего
    with zona.zone_ctx("wf"):
        sek.proverit_sektory_wf(real)
        for stroka, tekst in (("SBER,x,MOEXTN,u,d\n", "неприменима"), ("ZZZZ,x,MOEXFN,u,d\n", "нет в таблице")):
            bad = tmp_path / "bad.csv"
            bad.write_text("ticker,sektor_moex,indeks,source_url,accessed\n" + stroka)
            otkaz(ValueError, lambda: sek.proverit_sektory_wf(bad), tekst)


def test_sektor_map_v_wf_proveryaet_kartu(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,sektor_moex,indeks,source_url,accessed\nSBER,x,MOEXTN,u,d\n")
    assert sek.sektor_map(bad, {"MOEXTN"}) == {"SBER": "MOEXTN"}         # dev: прежнее
    with zona.zone_ctx("wf"), pytest.raises(ValueError):
        sek.sektor_map(bad, {"MOEXTN"})
