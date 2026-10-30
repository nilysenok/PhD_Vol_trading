"""Полная таблица голых идей (Ж-153 п. 1): известный ответ на синтетических таблицах."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7_polnaya as gp  # noqa: E402


def sd(vhod, storona, do, posle, vyhod="2016-03-01", sid="S10", var="centr", dn=2, sila=1.0):
    return {"sid": sid, "variant": var, "data_vhoda": vhod, "data_vyhoda": vyhod, "uderzhanie_dn": str(dn),
            "storona": str(storona), "sila": str(sila), "pnl_do_izderzhek": str(do), "pnl_posle": str(posle)}


def test_progrev_isklyuchaetsya_po_period():
    rows = [sd("2015-12-31", 1, 0.5, 0.4), sd("2016-01-01", 1, 0.1, 0.1)]
    assert [r["data_vhoda"] for r in gp.bez_progreva(rows, "train4")] == ["2016-01-01"]
    assert len(gp.bez_progreva(rows, "v2")) == 2  # В2: прогрев только < 2015
    assert len(gp.bez_progreva([sd("2014-12-31", 1, 1, 1)], "v2")) == 0


def test_storony_izvestnyj_otvet():
    rows = [sd("2016-02-01", 1, 0.04, 0.03), sd("2016-02-02", 1, 0.02, 0.01), sd("2016-02-03", -1, -0.01, -0.02),
            sd("2015-05-01", -1, 9.0, 9.0)]  # прогрев не считается
    s = gp.po_storonam(gp.bez_progreva(rows, "train4"), gody=2.0)
    assert s["n_long"] == 2 and s["n_short"] == 1
    assert s["dolya_long"] == pytest.approx(2 / 3) and s["dolya_short"] == pytest.approx(1 / 3)
    assert s["pnl_do_long"] == pytest.approx(3.0)  # (0.04+0.02)/2 лет, в %
    assert s["pnl_posle_long"] == pytest.approx(2.0)  # (0.03+0.01)/2 лет
    assert s["pnl_do_short"] == pytest.approx(-0.5) and s["pnl_posle_short"] == pytest.approx(-1.0)


def test_storony_pustaya_lenta():
    s = gp.po_storonam([], gody=4.0)
    assert s["n_long"] == 0 and s["dolya_long"] == 0.0 and s["pnl_do_short"] == 0.0


def test_p_perm_formula_i_sverka():
    assert gp.p_perm(1, 20) == pytest.approx(2 / 21)
    assert gp.p_perm(0, 20) == pytest.approx(1 / 21)
    gp.sverit_p({"dolya_ne_huzhe_centra": "0.05", "p_perm": str(2 / 21), "n": "20"})
    with pytest.raises(ValueError):  # k перепутан: p не сходится с долей
        gp.sverit_p({"dolya_ne_huzhe_centra": "0.10", "p_perm": str(2 / 21), "n": "20"})


def test_kontrol_stroka():
    r = {"strategiya": "S8", "centr": "0.3", "mediana": "-0.3", "min": "-0.7", "max": "2.2",
         "dolya_ne_huzhe_centra": "0.25", "p_perm": str(6 / 21), "n": "20"}
    k = gp.kontrol(r)
    assert k == {"med": -0.3, "min": -0.7, "max": 2.2, "p": pytest.approx(6 / 21), "n": 20}


def test_dnevnoj_ryad_i_sharpe_s_nulevymi_dnyami():
    rows = [sd("2016-02-01", 1, 0, 0.02, vyhod="2016-02-02"), sd("2016-02-01", -1, 0, 0.01, vyhod="2016-02-02"),
            sd("2016-02-02", 1, 0, -0.01, vyhod="2016-02-03")]
    cal = ["2016-02-01", "2016-02-02", "2016-02-03", "2016-02-04"]
    assert list(gp.dnevnoj_ryad(rows, cal)) == pytest.approx([0.0, 0.03, -0.01, 0.0])
    assert gp.sharpe_prib([0.0] * 5) == 0.0  # нулевой разброс
    assert gp.sharpe_prib([]) == 0.0  # пустая лента


def test_vne_oblasti_kf_po_nan_msr():
    assert gp.vne_oblasti("nan") == "да" and gp.vne_oblasti("") == "да"
    assert gp.vne_oblasti("-0.93") == "нет" and gp.vne_oblasti("0.5") == "нет"


def test_ekspoziciya_bez_sily_ravnyj_ves():
    r = sd("2016-02-01", 1, 0, 0, dn=17)
    r["sila"] = ""
    assert gp.ekspoziciya([r], dni=1) == pytest.approx(1.0)


def test_vyvod_po_zh153_dosl():
    assert gp.vyvod("S7", 0.14285714285714285) == "не включено: S7, p 0,143"
    assert gp.vyvod("S8", 0.0476) == "включено в v7 (Ж-153), низкий приоритет"
    assert gp.vyvod("S9", 0.0476) == "включено в v7 (Ж-153), низкий приоритет"
    for s in ("S10", "S11", "S12"):
        assert gp.vyvod(s, 0.1) == "включено в v7 (Ж-153)"


def test_ekspoziciya_valovaya_dolya_kapitala():
    # лонг 2 дня sila 1 и шорт 3 дня sila 0,5; 17 тикеров; окно 10 дней → (2·1 + 3·0,5)/17/10
    rows = [sd("2016-02-01", 1, 0, 0, dn=2, sila=1.0), sd("2016-02-01", -1, 0, 0, dn=3, sila=0.5)]
    assert gp.ekspoziciya(rows, dni=10) == pytest.approx(3.5 / 17 / 10)
    assert gp.ekspoziciya([], dni=10) == 0.0
