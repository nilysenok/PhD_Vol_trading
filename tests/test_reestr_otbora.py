"""Реестр отбора A: правила 19:35 (активность, плато), только-дополнение. Мутации — в самих тестах."""
import json
import os

import pytest

from fastbt import reestr_otbora as R

BAZA = 8.0  # сторон/год базы tune 2018


def tochka(storon, sos, plato=0.5):
    return {"storon": storon, "sos": sos, "plato": plato}


def test_odna_storona_v_god_vyrozhdena():
    o = R.ocenit(tochka(1.0, 24), BAZA, kandidat=True)
    assert o["status"] == "otbroshena_vyrozhdena" and o["aktivnost_ok"] == "0"
    assert o["vyrozhdena"] == "1"  # даже топ-3 по плато не спасает


def test_dva_soseda_plato_ne_ok():
    o = R.ocenit(tochka(8.0, 2), BAZA, kandidat=True)
    assert o["plato_ok"] == "0" and o["status"] == "otbroshena_vyrozhdena"
    assert R.ocenit(tochka(8.0, 3), BAZA, kandidat=True)["plato_ok"] == "1"


def test_granica_aktivnosti_vklyuchena():
    assert R.ocenit(tochka(4.0, 5), BAZA, False)["aktivnost_ok"] == "1"  # ровно 0,5 базы
    assert R.ocenit(tochka(3.99, 5), BAZA, False)["aktivnost_ok"] == "0"


def test_kandidat_i_rang():
    assert R.ocenit(tochka(8.0, 20), BAZA, True)["status"] == "kandidat_zamorozki"
    assert R.ocenit(tochka(8.0, 20), BAZA, False)["status"] == "otbroshena_rang"


def test_ne_izmereno_ne_kandidat_i_ne_vyrozhdena():
    o = R.ocenit({"storon": None, "sos": None}, BAZA, kandidat=True)
    assert o["status"] == "otbroshena_rang" and o["vyrozhdena"] == "0"
    assert o["aktivnost_ok"] == "" and o["plato_ok"] == ""


def test_mutaciya_porog_aktivnosti_nol_lovitsya():
    # без правила активности 1 сторона/год проходит — тест выше (odna_storona) обязан это ловить
    o = R.ocenit(tochka(1.0, 24), BAZA, True, porog=0.0)
    assert o["status"] != "otbroshena_vyrozhdena"


def test_mutaciya_min_sosedey_dva_lovitsya():
    o = R.ocenit(tochka(8.0, 2), BAZA, True, min_sos=2)
    assert o["plato_ok"] == "1"  # мутация меняет ответ → test_dva_soseda падает на ней


def stroka(k, status="otbroshena_rang"):
    return {"strategiya": "S1", "istochnik": "sweep_e1v2", "klyuch": k, "status": status}


def test_sliyanie_tolko_dopolnyaet():
    star = [stroka("a", "kandidat_zamorozki"), stroka("b")]
    nov = [stroka("a", "otbroshena_rang"), stroka("c")]
    res = R.sliyanie(star, nov)
    assert res[:2] == star  # старые на месте и не изменены
    assert [r["klyuch"] for r in res] == ["a", "b", "c"]
    assert R.sliyanie(res, nov) == res  # повтор ничего не меняет


def test_sliyanie_stroka_ne_propadaet_esli_ischezla_iz_novyh():
    star = [stroka("a"), stroka("b")]
    assert len(R.sliyanie(star, [stroka("a")])) == 2


def test_mutaciya_sliyanie_s_perezapisyu_lovitsya():
    def plokhoe(star, nov):  # мутация: реестр пересобирается заново
        return nov
    star, nov = [stroka("a"), stroka("b")], [stroka("a")]
    assert len(plokhoe(star, nov)) < len(star) <= len(R.sliyanie(star, nov))


def test_zapis_i_povtornaya_sborka(tmp_path):
    p = str(tmp_path / "r.csv")
    R.zapisat([dict(stroka("a"), tune_004="1.5")], p)
    sush = R.chitat(p)
    assert open(p, encoding="utf-8").readline().startswith("# реестр только дополняется")
    R.zapisat(R.sliyanie(sush, [stroka("a"), stroka("z")]), p)
    assert [r["klyuch"] for r in R.chitat(p)] == ["a", "z"] and R.chitat(p)[0]["tune_004"] == "1.5"


def test_norm_bool_i_none():
    assert R.norm("False") == R.norm(0.0) == 0.0 and R.norm("") is None and R.norm("atr14") == "atr14"


# --- на реальных таблицах ветки (train 2018–2019, без сети) ---
@pytest.fixture(scope="module")
def reestr():
    return R.chitat(R.VYHOD)


def test_vse_proby_ledger_v_reestre(reestr):
    for s in R.STRATEGII:
        for istochnik, papka in (("sweep_e1v2", R.NOVYE), ("sweep_e1", R.STARYE)):
            led = json.load(open(f"{papka}/_ledger/{s}/ledger.json", encoding="utf-8"))
            ist = (istochnik, "sweep_e1v2_dostroyka") if istochnik == "sweep_e1v2" else (istochnik,)     # достроенные соседи — отдельный источник
            ks = {r["klyuch"] for r in reestr if r["strategiya"] == s and r["istochnik"] in ist}
            assert ks == {R.klyuch_str(R.kanon_ledger(k)) for k in led}


def test_starye_proby_nedeystvitelny(reestr):
    star = [r for r in reestr if r["istochnik"] == "sweep_e1"]
    assert {r["status"] for r in star} == {R.STATUS_STARYJ}
    assert all("двойной знак, #326/#328" in r["prichina"] for r in star)
    assert all(r["status"] != R.STATUS_STARYJ and r["ekviv_s9"] == "ozhidaet"
               for r in reestr if r["istochnik"] == "sweep_e1v2")


def test_starye_tablicy_tozhe_v_reestre(reestr):
    for s in R.STRATEGII:
        ks = {r["klyuch"] for r in reestr if r["strategiya"] == s and r["istochnik"] == "sweep_e1"}
        assert {R.klyuch_str(k) for k in R.sobrat_tablicy(s, R.STARYE)} <= ks
    assert any(r["rezhim"] not in ("ledger",) for r in reestr if r["istochnik"] == "sweep_e1")


def test_staryj_klass_ne_byvaet_kandidatom(reestr):
    assert not [r for r in reestr if r["istochnik"] == "sweep_e1" and r["status"] == "kandidat_zamorozki"]
    for s in R.STRATEGII:
        assert all(r["status"] == R.STATUS_STARYJ for r in R.stroki_starye(s))


def test_mutaciya_starye_cherez_ocenit_dayut_kandidata():
    # мутация: старую пробу оценили бы по правилам v2 — стала бы кандидатом; тест выше такое ловит
    assert R.ocenit(tochka(8.0, 20), BAZA, True)["status"] == "kandidat_zamorozki" != R.STATUS_STARYJ


def test_kandidaty_proshli_pravila_i_net_dublej(reestr):
    kand = [r for r in reestr if r["status"] == "kandidat_zamorozki" and r["istochnik"] == "sweep_e1v2"]     # перезаморозка 2 — источник zamorozka_2
    assert len(kand) == 3 * len(R.STRATEGII)
    assert all(r["aktivnost_ok"] == "1" and r["plato_ok"] == "1" and r["vyrozhdena"] == "0" for r in kand)
    ks = [R.kluch_strok(r) for r in reestr]
    assert len(ks) == len(set(ks))


def test_vyrozhdennye_soglasovany_s_flagami(reestr):
    for r in reestr:
        if r["istochnik"] == "sweep_e1v2":
            assert (r["vyrozhdena"] == "1") == ("0" in (r["aktivnost_ok"], r["plato_ok"]))
            if r["vyrozhdena"] == "1":
                assert r["status"] == "otbroshena_vyrozhdena"


def test_svodka_soglasovana(reestr):
    sv = R.svodka(reestr)
    assert all(v["kandidatov"] == 3 and v["vyrozhdeno"] > 0 for v in sv.values())
    assert os.path.exists(R.VYHOD)
