"""Ж-150: разложение издержек. Известный ответ на синтетическом журнале + мутации (день d в ожидаемом PnL; доля от нетто)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import izderzhki_zh150 as iz


def _df(rows) -> pd.DataFrame:
    """rows: (вход, выход, удержание, терциль, gross, kom, short)."""
    r = [dict(src="golaya", sid="S7", tochka="", variant="centr", tip_vhoda="рынок", tf="", tiker="X", vhod=a, vyhod=b, uderzhanie=h, tercil=t, gross=g,
              kom=k, short=s, stavka=0.0, net=g - k - s) for a, b, h, t, g, k, s in rows]
    df = pd.DataFrame(r)
    df["vhod"], df["vyhod"], df["korz"] = pd.to_datetime(df.vhod), pd.to_datetime(df.vyhod), df.uderzhanie.map(iz.korzina)
    return df


SINT = [("2016-02-01", "2016-02-02", 1, 1, 1.0, 0.2, 0.0), ("2016-03-01", "2016-03-02", 1, 1, 1.0, 0.2, 0.2),
        ("2016-04-01", "2016-04-06", 5, 2, 4.0, 0.2, 0.0), ("2016-05-02", "2016-05-14", 12, 3, 10.0, 0.2, 0.0)]


def test_korzina_granicy():
    assert [iz.korzina(h) for h in (1, 2, 3, 4, 10, 11)] == ["1", "2–3", "2–3", "4–10", "4–10", ">10"]


def test_dolya_izvestnyj_otvet_i_mutaciya_ot_netto():
    assert iz.dolya(0.2, 0.2, 1.0) == 0.4                       # (0,2+0,2)/1,0 от валового
    assert abs(iz.dolya(0.2, 0.2, 1.0, baza="netto") - 0.4 / 0.6) < 1e-12
    assert iz.dolya(0.2, 0.2, 1.0) != iz.dolya(0.2, 0.2, 1.0, baza="netto")   # мутация «от нетто» ловится
    assert np.isnan(iz.dolya(0.1, 0.0, -1.0)) and np.isnan(iz.dolya(0.1, 0.0, 0.0))   # валовой ≤ 0 — не определено


def test_razrez_uderzhanie_i_tercil_izvestnyj_otvet():
    df = _df(SINT)
    t = iz.tablica(df)
    u = t[t.razdel == "2_uderzhanie"].set_index("srez")
    assert abs(u.loc["1", "dolya_izd"] - 0.3) < 1e-12 and not u.loc["1", "dolya_lt30"]   # 0,6 / 2,0: ровно 30 % — не «< 30 %»
    assert abs(u.loc["4–10", "dolya_izd"] - 0.05) < 1e-12 and bool(u.loc["4–10", "dolya_lt30"])
    assert abs(u.loc[">10", "dolya_izd"] - 0.02) < 1e-12 and "2–3" not in u.index
    c = t[t.razdel == "3_tercil"].set_index("srez")
    assert abs(c.loc["1", "dolya_izd"] - 0.3) < 1e-12 and abs(c.loc["2", "dolya_izd"] - 0.05) < 1e-12
    s = t[t.razdel == "1_sk_tip"].iloc[0]
    assert s.n == 4 and abs(s.gross_sr - 4.0) < 1e-12 and abs(s.kom_sr - 0.2) < 1e-12 and abs(s.short_sr - 0.05) < 1e-12
    assert abs(s.dolya_izd - 1.0 / 16.0) < 1e-12 and abs(s.net_sr - (16 - 1.0) / 4) < 1e-12


def test_itogovaya_stroka():
    assert iz.itog(_df(SINT)) == ("издержки съедают 6 % валового; окупаются сделки с удержанием ≥ 4 дней и рангом ≥ 2 "
                                  "(терциль; доля издержек < 30 % валового)")


def test_itog_bez_valovogo():
    assert "не определено" in iz.itog(_df([("2016-02-01", "2016-02-02", 1, 1, -1.0, 0.2, 0.0)]))


def test_sharpe_do_i_posle_rjadom_posle_men_she():
    t = iz.tablica(_df(SINT))
    s = t[t.razdel == "1_sk_tip"].iloc[0]
    assert s.sharpe_do > s.sharpe_posle > 0


def _istoriya() -> pd.DataFrame:
    """40 сделок ячейки (терциль 1, удержание 2): gross 1,0, издержки 0,1; плюс «плохая» сделка, закрытая в день d = 2016-06-01."""
    dni = pd.bdate_range("2016-01-04", periods=40)
    rows = [(a, a + pd.Timedelta(days=2), 2, 1, 1.0, 0.1, 0.0) for a in dni]
    rows += [("2016-05-30", "2016-06-01", 2, 1, -1000.0, 0.1, 0.0), ("2016-06-01", "2016-06-03", 2, 1, 1.0, 0.1, 0.0)]
    return _df(rows)


def test_ozhid_bez_dnya_d_i_mutaciya_s_dnyom_d():
    g = _istoriya()
    e1 = iz.ozhid(g, "2016-06-01", 1, "2–3", lag=1)
    e0 = iz.ozhid(g, "2016-06-01", 1, "2–3", lag=0)                    # мутация: сделка, закрытая в день d, видна
    assert abs(e1[0] - 1.0) < 1e-12 and e1[2] == 40
    assert e0[2] == 41 and e0[0] < 0                                   # мутация ловится: ожидание ломается
    assert iz.ozhid(g, "2016-06-01", 1, "2–3", lag=1)[0] != e0[0]


def test_k_okup_probe_prohodit_bez_dnya_d_i_ne_prohodit_s_nim():
    g = _istoriya()
    ok1, ok0 = iz.k_okup(g, 5, lag=1)[0], iz.k_okup(g, 5, lag=0)[0]
    assert ok1["n"] == 42 and ok1["n_proshlo"] - ok0["n_proshlo"] == 1   # 1,0 ≥ 5·0,1 для сделки 06-01; при lag=0 ожидание < 0 — единственное различие
    assert ok1["n_proshlo"] > 0


def test_ozhid_malo_nablyudenij_net_ocenki():
    assert np.isnan(iz.ozhid(_istoriya().head(10), "2016-06-01", 1, "2–3")[0])


def test_chitat_golaya_otbrasyvaet_2020_i_2015(tmp_path):
    p = tmp_path / "s.csv"
    h = "sid,variant,tiker,data_vhoda,data_vyhoda,uderzhanie_dn,storona,tip_vhoda,tf,vhod,sila,tercil_sily,pnl_do_izderzhek,komissiya,plata_za_short,stavka,pnl_posle\n"
    r = lambda a, b: f"S7,centr,X,{a},{b},1,1,рынок,,d1,1.0,3,0.01,0.001,0.0,0.0,0.009\n"   # noqa: E731
    p.write_text("# {}\n" + h + r("2015-12-30", "2015-12-31") + r("2016-02-01", "2016-02-02") + r("2019-12-30", "2020-01-02") + r("2020-03-02", "2020-03-03"), encoding="utf-8")
    iz.OTBROSHENO.clear()
    df = iz.chitat_golaya(p)
    assert len(df) == 1 and df.vhod.iloc[0] == pd.Timestamp("2016-02-01") and iz.OTBROSHENO["golaya"] == 3
    assert abs(df.gross.iloc[0] - 1.0) < 1e-12                         # доли капитала → %
