"""К0-д Ж-051: критерий §3 (граница 0,10), правило остановки Ж-054, планы итераций. Синтетика, без данных.
Ledger, общий PBO и драйвер на подменённом оценщике — `test_zh051_iter_drv.py`."""
import pandas as pd
import pytest
import yaml

from fastbt import zh051_iter as I, zh051_iter10 as I10, zh051_kriteriy as K

OK = dict(tsel_pi=0.60, tsel_p1=0.50, msr_pi={2018: 0.55, 2019: 0.9}, msr_p1={2018: 0.5, 2019: 0.8}, nb_tsel=[0.6, 0.6, 0.1],
          sdelok_pi={2018: 40, 2019: 40}, s9_ok=True, g4_ok=True, pbo_do=0.5, pbo_posle=0.5, kontrol=[0.4] * 20)


def kr(**kw):
    return K.kriteriy({**OK, **kw})


def test_kriteriy_vse_ok_i_granitsa_delta():
    assert kr()["proshla"]
    assert kr(tsel_pi=0.5999)["P1"] is False                      # 0,0999 < 0,10
    assert kr(tsel_pi=0.1 + 0.3, tsel_p1=0.3)["P1"]               # ровно 0,10 на плавающей точке — проходит
    assert kr(tsel_pi=float("nan"))["proshla"] is False


def test_kriteriy_p2_po_godam_p3_p4():
    assert kr(msr_pi={2018: 0.39, 2019: 0.9})["P2"] is False and kr(msr_pi={2018: 0.4, 2019: 0.9})["P2"]
    assert kr(msr_pi={2018: 0.9, 2019: 0.69})["P2"] is False        # год нельзя отдать
    assert kr(nb_tsel=[0.6, 0.4, 0.4])["P3"] is False and kr(nb_tsel=[0.6, 0.7])["P3"]
    assert kr(nb_tsel=[0.6])["P3"] and kr(nb_tsel=[float("nan"), 0.7, 0.2])["P3"] is False
    assert kr(sdelok_pi={2018: 29, 2019: 40})["P4"] is False
    assert kr(pbo_posle=0.53)["P4"] and kr(pbo_posle=0.54)["P4"] is False       # +0,03 — граница
    assert kr(s9_ok=False)["P4"] is False and kr(g4_ok=False)["P4"] is False


def test_kriteriy_p5_kontrol_19_iz_20():
    c = lambda n_bad: [0.5] * (20 - n_bad) + [0.8] * n_bad                       # noqa: E731  контроль выше Pi — «плохой»
    assert kr(kontrol=c(0))["P5"] and kr(kontrol=c(1))["P5"]                      # 19 из 20, p = 2/21 ≤ 0,1
    assert kr(kontrol=c(1))["p_kontrol"] == pytest.approx(2 / 21)
    assert kr(kontrol=c(2))["P5"] is False                                        # 18 из 20
    assert kr(kontrol=[0.4] * 19)["P5"] is False                                  # меньше 20 контролей
    assert kr(kontrol=None)["P5"] is True


def test_ostanovka_zh054():
    s = K.ostanovka({"S1": .41, "S2": .8, "S3": .21, "S4": .7, "S5": .5, "S6": .6}, {})
    assert s["stop"] and s["itog"] == "остановка, A′ = A"
    s = K.ostanovka({"S1": .2, "S2": .8, "S3": .6, "S4": .7, "S5": .5, "S6": .6}, {"S1": True})   # ровно 0,2 — не «> 0,2»
    assert not s["stop"] and s["dalshe"] == ["S2", "S3", "S4", "S5", "S6"]
    with pytest.raises(ValueError):
        K.ostanovka({"S1": float("nan"), "S2": .8}, {})


def test_elem_it2_realnaya_stroka_S1_ne_proshla():
    r = pd.read_csv("docs/tables/zh051/it2/S1_itog.csv", comment="#").iloc[-1].to_dict()
    c = K.elem_it2(r)
    assert c["P1"] and c["P4"] is False and not c["proshla"]      # G4_osi_centra = False в таблице #359


def test_plan_chislo_tochek_iz_yaml():
    for it, n in ((3, 4), (4, 60)):
        y = yaml.safe_load(open(f"fastbt/designs/sweep/zh051_it{it}.yaml", encoding="utf-8"))
        pts, grid = I.plan(y)
        assert len(pts) == n == y["N"]["S1"]
    assert I.norm({"top_k": "net", "sektor_trend": "net", "shirina": "net", "ma_n": 50, "porog_shiriny": 0.4}) is None
    assert I.norm({"sektor_trend": "po_trendu", "shirina": "net", "ma_n": 50, "porog_shiriny": 0.4})["porog_shiriny"] == 0.5


def test_kombinacii_iteracii_10():
    E = ["ans", "top_k", "sektor"]
    assert len(I10.kombinacii(E)) == 4 and I10.kombinacii(["ans"]) == [] and I10.kombinacii(E[:2]) == [("ans", "top_k")]
    assert I10.sosedi(("ans", "top_k"), E) == [("top_k",), ("ans",), ("ans", "top_k", "sektor")]       # Хэмминг 1
