"""wf-L7: корректировка цен по годным строкам свода Д8 (2022+); ≤ 2021 — побайтно прежнее. Синтетика, без сети.

Календарь подменён (рабочие дни 2014-06 … 2024-12); свод — временный csv. Мутация: «берётся вся таблица 2022+»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy as dv
from fastbt import zona

CAL = pd.bdate_range("2014-06-02", "2024-12-31").values.astype("datetime64[ns]")
COLS = "ticker,record_date,dps_rub,announce_pit,godna,prichina\n"


@pytest.fixture
def svod(tmp_path, monkeypatch):
    p = tmp_path / "svod.csv"
    p.write_text(COLS
                 + "SBER,2023-08-10,20.0,2023-05-01,1,\n"        # годная 2023
                 + "SBER,2023-11-20,15.0,2023-10-01,0,net_istochnika\n"   # негодная 2023
                 + "SBER,2021-12-01,5.0,2021-10-01,1,\n"         # ≤ 2021: из свода не берётся (там прежние таблицы)
                 + "SBER,2024-05-10,0.0,2024-04-01,1,\n")        # нулевая выплата — не событие
    monkeypatch.setattr(dv, "SVOD", p)
    monkeypatch.setattr(dv, "_calendar", lambda: CAL)
    monkeypatch.delenv("FASTBT_Z2_DIV", raising=False)
    return p


@pytest.fixture
def svod_z(svod, monkeypatch):
    """Свод + режим FASTBT_Z2_DIV=SVOD (вне dev)."""
    monkeypatch.setenv("FASTBT_Z2_DIV", dv.MARKER_SVOD)
    return svod


def _gaps(dates=CAL):
    return dv.gap_divs("SBER", dates)


def _i(d):
    return int(np.searchsorted(CAL, np.datetime64(d)))


def _inv_negodnaya_ne_koryectiruet():
    d = _gaps()
    assert d[_i("2023-11-21")] == 0.0 and d[_i("2023-11-20")] == 0.0     # негодная 2023-11-20 (cum 2023-11-17, гэп 11-20)
    assert d[_i("2023-11-17")] == 0.0 and d[_i("2023-11-20") + 1] == 0.0
    assert d[_i("2022-01-01"):].sum() == pytest.approx(20.0)               # с 2022 — только годная


def test_godnaya_vyplata_2023_koryectiruet_ryad(svod_z):
    with zona.zone_ctx("test1"):
        d = _gaps()
        g = _i("2023-08-10")           # реестр 10.08, T+1 с 31.07.2023: cum 09.08, гэп 10.08
        assert d[g] == 20.0 and d[_i("2022-01-01"):].sum() == pytest.approx(20.0)
        close = np.full(len(CAL), 100.0)
        (adj,) = dv.adjust_ohlc("SBER", CAL, close, close.copy())
        assert adj[g] / adj[g - 1] == pytest.approx(1 / 0.8)                  # D/c = 0,2: ÷ (1 − 0,2)
        assert adj[g + 1] == adj[g] and adj[g - 2] == adj[g - 1]              # до и после гэпа уровень ровный
        assert adj[0] == 100.0                                                # до первой выплаты ряд сырой


def test_negodnaya_ne_koryectiruet(svod_z):
    with zona.zone_ctx("test1"):
        _inv_negodnaya_ne_koryectiruet()


def test_mutaciya_vsya_tablica_2022_lovitsya(svod_z, monkeypatch):
    """Мутация: читаются все строки 2022+ (godna игнорируется) — инвариант негодной обязан падать."""
    def vsya(ticker):
        s = pd.read_csv(dv.SVOD)
        s = s[(s["ticker"] == ticker) & (s["dps_rub"] > 0) & (pd.to_datetime(s["record_date"]) >= pd.Timestamp(dv.SVOD_OT))]
        return pd.DataFrame({"last_cum_date": dv.last_cum(pd.to_datetime(s["record_date"])), "dps_rub": s["dps_rub"]})
    monkeypatch.setattr(dv, "_read_svod", vsya)
    with zona.zone_ctx("test1"), pytest.raises(AssertionError):
        _inv_negodnaya_ne_koryectiruet()


def test_dev_svod_ne_chitaetsya_train4_pobaitno(svod, monkeypatch):
    """dev (≤ 2021): свод не подключается ни к событиям, ни к ключу кэша; события = прежние таблицы."""
    assert zona.name() == "dev" and not dv._svod_on()
    e = dv.events("SBER")
    ref = pd.concat([dv._read(p, "SBER") for p in (dv.TABLE, dv.TABLE_2015)], ignore_index=True)
    ref = ref.sort_values("last_cum_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(e, ref)
    assert (e["last_cum_date"] < pd.Timestamp(dv.SVOD_OT)).all()
    k = dv.digest()
    monkeypatch.setattr(dv, "SVOD", dv.SVOD.with_name("net_takogo.csv"))
    assert dv.digest() == k                                              # dev-ключ от свода не зависит


def test_do_2022_v_zone_so_svodom_to_zhe_chto_bez_svoda(svod_z, monkeypatch):
    """Зона со сводом: строки реестра < 2022 те же, что без свода (выплата SBER 2021-12-01 из свода не удваивает таблицу)."""
    with zona.zone_ctx("test1"):
        a = dv.events("SBER")
        a = a[a["last_cum_date"] < pd.Timestamp(dv.SVOD_OT)].reset_index(drop=True)
    ref = pd.concat([dv._read(p, "SBER") for p in (dv.TABLE, dv.TABLE_2015)], ignore_index=True)
    ref = ref.sort_values("last_cum_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, ref)


def test_svod_v_klyuche_kesha_v_zone(svod_z, monkeypatch, tmp_path):
    with zona.zone_ctx("test1"):
        k = dv.digest()
        p2 = tmp_path / "svod2.csv"
        p2.write_text(COLS + "SBER,2023-08-10,20.0,2023-05-01,0,net_istochnika\n")
        monkeypatch.setattr(dv, "SVOD", p2)
        assert dv.digest() != k


def test_marker_svod_v_dev_i_bez_fajla_otkaz(svod, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTBT_Z2_DIV", dv.MARKER_SVOD)
    with pytest.raises(ValueError):
        dv.events("SBER")                                                # dev + SVOD — молча не проходит
    monkeypatch.setattr(dv, "SVOD", tmp_path / "net.csv")
    with zona.zone_ctx("test1"), pytest.raises(FileNotFoundError):
        dv.events("SBER")                                                # свода нет — отказ, не тихое «без дивидендов»


def test_reestr_posle_poslednego_bara_propuskaetsya(svod_z, monkeypatch):
    """Реестр позже последнего бара ряда — без календаря last_cum был бы неверен; строка пропускается."""
    svod_z.write_text(COLS + "SBER,2025-03-03,9.0,2025-01-01,1,\n")
    with zona.zone_ctx("test1"):
        assert dv.events("SBER")["dps_rub"].tolist().count(9.0) == 0
