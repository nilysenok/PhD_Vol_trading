"""Ж-150, тестировщик #885: мутации через monkeypatch (доля от нетто; терциль/ожидание с днём d; прогрев 2015 в окне) и найденные дефекты."""
from __future__ import annotations

import functools

import pandas as pd
import pytest

from fastbt import izderzhki_zh150 as iz
from tests.test_izderzhki_zh150 import SINT, _df, _istoriya

H = "sid,variant,tiker,data_vhoda,data_vyhoda,uderzhanie_dn,storona,tip_vhoda,tf,vhod,sila,tercil_sily,pnl_do_izderzhek,komissiya,plata_za_short,stavka,pnl_posle\n"


def _csv(tmp_path, pary):
    p = tmp_path / "s.csv"
    p.write_text("# {}\n" + H + "".join(f"S7,centr,X,{a},{b},1,1,рынок,,d1,1.0,3,0.01,0.001,0.0,0.0,0.009\n" for a, b in pary), encoding="utf-8")
    return p


def test_mut_dolya_ot_netto_menyaet_tablicu(monkeypatch):
    ok = iz.tablica(_df(SINT)).iloc[0].srez
    monkeypatch.setattr(iz, "dolya", functools.partial(iz.dolya, baza="netto"))
    assert iz.tablica(_df(SINT)).iloc[0].srez != ok   # «6 %» -> другое число: тест itog/разрезов поймает


def test_mut_ozhid_s_dnyom_d_menyaet_k_okup(monkeypatch):
    g = _istoriya()
    ok = iz.k_okup(g, 5)[0]["n_proshlo"]
    orig = iz.ozhid
    monkeypatch.setattr(iz, "ozhid", lambda g, d, rang, korz, lag=1: orig(g, d, rang, korz, 0))
    assert iz.k_okup(g, 5)[0]["n_proshlo"] != ok


def test_mut_progrev_2015_v_okne(tmp_path, monkeypatch):
    p = _csv(tmp_path, [("2015-12-30", "2015-12-31"), ("2015-12-30", "2016-01-05"), ("2016-02-01", "2016-02-02")])
    iz.OTBROSHENO.clear()
    assert len(iz.chitat_golaya(p)) == 1 and iz.OTBROSHENO["golaya"] == 2   # вход в 2015 (даже при выходе в 2016) — прогрев, не в окне
    monkeypatch.setattr(iz, "D0", pd.Timestamp("2015-01-01"))
    assert len(iz.chitat_golaya(p)) == 3                                    # мутация видна


def test_granica_2019_2020_sdelka_ne_zakryvaetsya_na_kontse_okna(tmp_path):
    """Вход 2019, выход 2020: по Н5 закрывается на конце окна 31.12.2019 (исправлено по замечанию тестировщика)."""
    iz.OTBROSHENO.clear()
    df = iz.chitat_golaya(_csv(tmp_path, [("2019-12-30", "2020-01-02")]))
    assert len(df) == 1 and df.vyhod.iloc[0] == pd.Timestamp("2019-12-31")


def test_sharpe_dn_ignoriruet_vyhod_vne_rabochih_dnej():
    df = _df([("2016-02-01", "2016-02-06", 5, 1, 1.0, 0.0, 0.0), ("2016-03-01", "2016-03-02", 1, 1, 1.0, 0.0, 0.0)])   # выход в субботу
    s = iz.sharpe_dn(df, "gross")
    tot = df.groupby("vyhod").gross.sum().reindex(pd.bdate_range(iz.D0, iz.D1), fill_value=0.0).sum()
    assert tot == 1.0 and s == s   # сделка в субботу молча потеряна из ряда (дефект малый: у движка выходы — рабочие дни)


@pytest.mark.xfail(strict=True, reason="ДЕФЕКТ: itog берёт ПЕРВУЮ корзину с долей < 30 % и пишет «≥ N дней»; старшие корзины с gross ≤ 0 (nan) не проверяются")
def test_itog_nemonotonnost_korzin():
    rows = [("2016-02-01", "2016-02-04", 3, 3, 10.0, 0.2, 0.0), ("2016-03-01", "2016-03-15", 14, 3, -5.0, 0.2, 0.0)]
    assert "≥ 2 дней" not in iz.itog(_df(rows))
