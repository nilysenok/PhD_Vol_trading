"""wf-L7: корректировка цен 2022+ по строкам свода Д8 с godna_korr=1 (зона wf, автоматически); ≤ 2021 — побайтно прежнее.

Синтетика, без сети. Календарь подменён (рабочие дни 2014-06 … 2024-12); свод — временный csv.
Мутации: «берётся вся таблица 2022+», «берётся godna вместо godna_korr»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy as dv
from fastbt import zona

CAL = pd.bdate_range("2014-06-02", "2024-12-31").values.astype("datetime64[ns]")
COLS = "ticker,record_date,dps_rub,announce_pit,godna,godna_korr,prichina\n"


@pytest.fixture
def svod(tmp_path, monkeypatch):
    p = tmp_path / "svod.csv"
    p.write_text(COLS
                 + "SBER,2023-08-10,20.0,,0,1,net_daty_obyavleniya\n"     # реестр и сумма подтверждены, даты объявления нет
                 + "SBER,2023-11-20,15.0,2023-10-01,1,0,\n"               # godna=1, но корректировке не годна (синтетика для мутации)
                 + "SBER,2024-02-12,7.0,,0,0,net_istochnika\n"            # негодна во всём
                 + "SBER,2021-12-01,5.0,2021-10-01,1,1,\n"                # ≤ 2021: из свода не берётся (там прежние таблицы)
                 + "SBER,2024-05-10,0.0,2024-04-01,1,1,\n")               # нулевая выплата — не событие
    monkeypatch.setattr(dv, "SVOD", p)
    monkeypatch.setattr(dv, "_calendar", lambda: CAL)
    monkeypatch.delenv("FASTBT_Z2_DIV", raising=False)
    return p


@pytest.fixture
def wf(svod):
    with zona.zone_ctx("wf"):
        yield svod


def _gaps():
    return dv.gap_divs("SBER", CAL)


def _i(d):
    return int(np.searchsorted(CAL, np.datetime64(d)))


def _inv_tolko_godna_korr():
    d = _gaps()
    assert d[_i("2023-11-20")] == 0.0 and d[_i("2024-02-12")] == 0.0       # godna_korr=0 не корректирует
    assert d[_i("2022-01-01"):].sum() == pytest.approx(20.0)               # с 2022 — только godna_korr=1


def test_vyplata_godna_korr_bez_godna_koryectiruet_ryad(wf):
    d = _gaps()
    g = _i("2023-08-10")               # реестр 10.08, T+1 с 31.07.2023: cum 09.08, гэп 10.08
    assert d[g] == 20.0
    _inv_tolko_godna_korr()
    close = np.full(len(CAL), 100.0)
    (adj,) = dv.adjust_ohlc("SBER", CAL, close, close.copy())
    assert adj[g] / adj[g - 1] == pytest.approx(1 / 0.8)                  # D/c = 0,2: ÷ (1 − 0,2)
    assert adj[g + 1] == adj[g] and adj[g - 2] == adj[g - 1]              # до и после гэпа уровень ровный
    assert adj[0] == 100.0                                                # до первой выплаты ряд сырой


def test_svod_vklyuchen_v_zone_wf_bez_peremennoj(wf):
    assert zona.name() == "wf" and dv._svod_on()


def test_mutaciya_godna_vmesto_godna_korr_lovitsya(wf, monkeypatch):
    def po_godna(ticker):
        s = pd.read_csv(dv.SVOD)
        s = s[(s["ticker"] == ticker) & (s["godna"] == 1) & (s["dps_rub"] > 0) & (pd.to_datetime(s["record_date"]) >= pd.Timestamp(dv.SVOD_OT))]
        return pd.DataFrame({"last_cum_date": dv.last_cum(pd.to_datetime(s["record_date"])), "dps_rub": s["dps_rub"]})
    monkeypatch.setattr(dv, "_read_svod", po_godna)
    with pytest.raises(AssertionError):
        _inv_tolko_godna_korr()


def test_mutaciya_vsya_tablica_2022_lovitsya(wf, monkeypatch):
    def vsya(ticker):
        s = pd.read_csv(dv.SVOD)
        s = s[(s["ticker"] == ticker) & (s["dps_rub"] > 0) & (pd.to_datetime(s["record_date"]) >= pd.Timestamp(dv.SVOD_OT))]
        return pd.DataFrame({"last_cum_date": dv.last_cum(pd.to_datetime(s["record_date"])), "dps_rub": s["dps_rub"]})
    monkeypatch.setattr(dv, "_read_svod", vsya)
    with pytest.raises(AssertionError):
        _inv_tolko_godna_korr()


def _ref_do_2022():
    ref = pd.concat([dv._read(p, "SBER") for p in (dv.TABLE, dv.TABLE_2015)], ignore_index=True)
    return ref.sort_values("last_cum_date").reset_index(drop=True)


def test_dev_svod_ne_chitaetsya_train4_pobaitno(svod, monkeypatch):
    """dev (≤ 2021): свод не подключается ни к событиям, ни к ключу кэша; события = прежние таблицы."""
    assert zona.name() == "dev" and not dv._svod_on()
    e = dv.events("SBER")
    pd.testing.assert_frame_equal(e, _ref_do_2022())
    assert (e["last_cum_date"] < pd.Timestamp(dv.SVOD_OT)).all()
    k = dv.digest()
    monkeypatch.setattr(dv, "SVOD", dv.SVOD.with_name("net_takogo.csv"))
    assert dv.digest() == k                                              # dev-ключ от свода не зависит


def test_do_2022_v_zone_wf_to_zhe_chto_v_dev(wf):
    """wf: события с реестром < 2022 те же, что в dev (27 выплат ≤ 2021 вне godna продолжают применяться)."""
    a = dv.events("SBER")
    a = a[a["last_cum_date"] < pd.Timestamp(dv.SVOD_OT)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, _ref_do_2022())


def test_test1_svod_ne_vklyuchaetsya(svod):
    """Прочие зоны — как было: test1 без FASTBT_Z2_DIV отказывает, свод не подхватывается молча."""
    with zona.zone_ctx("test1"):
        assert not dv._svod_on()
        with pytest.raises(FileNotFoundError):
            dv.events("SBER")


def test_svod_v_klyuche_kesha_v_zone_wf(wf, monkeypatch, tmp_path):
    k = dv.digest()
    p2 = tmp_path / "svod2.csv"
    p2.write_text(COLS + "SBER,2023-08-10,20.0,,0,0,net_istochnika\n")
    monkeypatch.setattr(dv, "SVOD", p2)
    assert dv.digest() != k


def test_net_fajla_i_net_kolonki_otkaz(svod, monkeypatch, tmp_path):
    with zona.zone_ctx("wf"):
        p = tmp_path / "staryj.csv"
        p.write_text("ticker,record_date,dps_rub,announce_pit,godna,prichina\nSBER,2023-08-10,20.0,2023-05-01,1,\n")
        monkeypatch.setattr(dv, "SVOD", p)
        with pytest.raises(ValueError, match="godna_korr"):
            dv.events("SBER")                                            # свод без godna_korr — не тихое «все годные»
        monkeypatch.setattr(dv, "SVOD", tmp_path / "net.csv")
        with pytest.raises(FileNotFoundError):
            dv.events("SBER")


def test_reestr_posle_poslednego_bara_propuskaetsya(wf):
    """Реестр позже последнего бара ряда — без календаря last_cum был бы неверен; строка пропускается."""
    wf.write_text(COLS + "SBER,2025-03-03,9.0,2025-01-01,1,1,\n")
    assert dv.events("SBER")["dps_rub"].tolist().count(9.0) == 0
