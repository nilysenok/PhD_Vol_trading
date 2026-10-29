"""Перезаморозка A′, вариант (а): фильтр «есть боевой эталон», плато на воспроизводимых точках, сверка 18 кандидатов с s9, даты, реестр.

Допуск сверки — как в `test_ekviv_sweep_s9.py` (объявлен до счёта): позиции побайтно, дневной PnL abs ≤ 1e-12, окна 2018 и 2019, комиссии 0,04 и 0,035 %.
Мутации: фильтр, пропускающий всё (маски/√RV/trail_n); плато с краями и дублями; позиции сдвинуты на бар; даты кандидата и базы разные; реестр без net_etalona.
Данные: validation 2020–2021 не читается. Запуск: FASTBT_MAIN=$PWD pytest tests/test_ekviv_kand.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import ekviv_filtr as ef
from fastbt import ekviv_kand as kn
from fastbt import ekviv_s9 as ek
from fastbt import reestr_otbora_pz as pz
from fastbt import s9, sweep_post
from fastbt import sweep_val_kand as kd

DATA_OK = (s9.OUT / "cache_latest_v2.txt").exists() and (Path(s9.MAIN) / "data" / "regen" / "features_pit_v3.parquet").exists()
real = pytest.mark.skipif(not DATA_OK, reason="нет данных проекта (кэш s9 v2, features_pit_v3)")


@pytest.fixture(autouse=True, scope="module")
def _s9_globali_vosstanovleny():
    """Р-6 (29.10, как mutations.md № 43): S9Baza → sweep_val_baza.s9_engine() делает s9.use_v2() — глобально и навсегда;
    после этого файла test_progon_urovni и test_rassl_s56_equiv искали cache_latest_v2_train4.txt и падали (в одиночку — зелёные)."""
    keys = ("V2", "TABLES", "SUFFIX", "COMM", "SCOPE", "YEARS", "VAL_START")
    old = {k: getattr(s9, k) for k in keys}
    src = dict(s9.SOURCES)
    yield
    for k, v in old.items():
        setattr(s9, k, v)
    s9.SOURCES.clear()
    s9.SOURCES.update(src)


PTS20 = [x for x in ek.load_points()["tochki"] if x["sid"] != "A"]
KAND = ek.load_points(kn.YAML)["tochki"]
KA = pd.read_csv("docs/tables/sweep_e1v2/kand_a.csv", comment="#")
CENTR = {x["sid"]: x["p"] for x in PTS20 if x["klass"] == "tsentr"}


# ---------- фильтр «есть боевой эталон» ----------

def razbor():
    """Расхождения фильтра с полем `etalon` 20 фиксированных точек (пусто = согласован)."""
    return [x["id"] for x in PTS20 if ef.voproizvodima(x["sid"], x["p"]) != (x["etalon"] == "da")]


def test_filtr_soglasovan_s_20_tochkami_s9():
    assert razbor() == []
    assert sorted(x["id"] for x in PTS20 if not ef.voproizvodima(x["sid"], x["p"])) == ["S3_v", "S4_v", "S5_v", "S6_v"]


@pytest.mark.parametrize("sid,izm,slovo", [
    ("S3", {"trail_n": 60}, "TREND_TN"), ("S3", {"trail_type": "off"}, "trail_type"), ("S4", {"rm_edinitsa": "sqrt_rv"}, "√RV"),
    ("S5", {"vf_porog": 0.75}, "vf_porog"), ("S1", {"q_sily": 0.5}, "q_sily"), ("S1", {"vol_regime_max": 1.0}, "vol_regime_max"),
    ("S3", {"obyom_mnozh_sma20": 1.5}, "obyom"), ("S6", {"rm_tip": "sqrt_rv"}, "√RV"), ("S3", {"adx_min": 25}, "adx_min")])
def test_filtr_otkazyvaet_bez_etalona(sid, izm, slovo):
    p = {**CENTR[sid], **izm}
    assert not ef.voproizvodima(sid, p) and slovo in ef.prichina_net(sid, p)
    assert ef.voproizvodima(sid, CENTR[sid])


def test_filtr_trail_n_v_spiske_dopuskaetsya():
    assert ef.voproizvodima("S3", {**CENTR["S3"], "trail_n": 30})


def test_mutatsiya_filtr_propuskaet_vse_lovitsya(monkeypatch):
    monkeypatch.setattr(ek, "MASK_BASE", {})                       # маски после кэша больше не ограничивают
    monkeypatch.setattr(ef, "TREND_TN", list(range(1, 300)))
    assert razbor() != []


# ---------- плато среди воспроизводимых: выбор топ-3 ----------

def _t(rows):
    return pd.DataFrame([dict(plateau=p, edge=e, tune_sharpe=a, ctrl_sharpe=b, sdelok_y=c) for p, e, a, b, c in rows])


def test_izbrat_bez_kraev_i_dublej_po_plato():
    t = _t([(0.9, True, 2, 1, 5), (0.8, False, 1, 1, 5), (0.7, False, 1, 1, 5), (0.6, False, 0.5, 0, 6), (0.5, False, 0.3, 0, 7), (0.4, False, 0.2, 0, 8)])
    assert list(ef.izbrat(t).index) == [1, 3, 4]                   # край 0 и дубль 2 (подпись как у 1) отброшены
    assert list(ef.izbrat(t, 2).index) == [1, 3]


def test_mutatsiya_izbrat_s_kraem_i_dublem_lovitsya(monkeypatch):
    t = _t([(0.9, True, 2, 1, 5), (0.8, False, 1, 1, 5), (0.7, False, 1, 1, 5), (0.6, False, 0.5, 0, 6)])
    ok = list(ef.izbrat(t).index)
    monkeypatch.setattr(ef, "_podpis", lambda r: r.name)           # дубли по подписи не сводятся
    assert list(ef.izbrat(t).index) != ok


# ---------- таблица кандидатов, yaml, фабрика ----------

def test_kand_a_top3_vosproizvodimy_i_pravila_plato():
    top = KA[KA["top3"] == 1]
    assert top.groupby("strategiya").size().to_dict() == {f"S{i}": 3 for i in range(1, 7)}
    assert (top["neighbors"] >= 3).all() and (KA["neighbors"] >= 3).all()
    for sid in kd.SIDS:
        for c in kd.kandidaty(sid)["cands"]:
            assert ef.voproizvodima(sid, c["point"]), (sid, c["rank"])


def test_kandidaty_fabriki_sovpadayut_s_flagom_top3():
    for sid in kd.SIDS:
        flag = KA[(KA["strategiya"] == sid) & (KA["top3"] == 1)]["rank"].tolist()
        assert [c["rank"] for c in kd.kandidaty(sid)["cands"]] == flag


def test_yaml_kandidatov_rovno_kandidaty_fabriki():
    assert len(KAND) == 18 and all(x["etalon"] == "da" and x["klass"] == "kandidat_zamorozki" for x in KAND)
    want = [(f"K_{s}_{i}", s, c["point"]) for s in kd.SIDS for i, c in enumerate(kd.kandidaty(s)["cands"], 1)]
    assert [(x["id"], x["sid"], x["p"]) for x in KAND] == want
    assert ek.load_points(kn.YAML)["dopusk"] == {"pozitsii": "pobaytno", "pnl_abs": 1e-12}


def test_stroka_sborki_posle_perezamorozki_tsentr_i_boevaya_ostayutsya():
    t = pd.read_csv("docs/tables/sweep_e1v2/A_sborki.csv", comment="#")
    assert (t["kind"] == "sborka").sum() == 36 and {"S_centr_ravnye", "A_boevaya_s9", "A_novye_ravnye"} <= set(t["kind"])
    assert sorted(t.loc[t["top2"] > 0, "top2"]) == [1, 2]


# ---------- сверка 18 кандидатов с боевым путём s9 ----------

@pytest.fixture(scope="module")
def sverka():
    from fastbt.sweep_sk import SkEvaluator
    evs, rows = {}, []
    for pt in KAND:
        rows += ek.run_sk(pt, evs.setdefault(pt["sid"], SkEvaluator(pt["sid"])), (0.0004, 0.00035))
    return pd.DataFrame(rows)


@real
def test_18_kandidatov_sovpali_s_s9(sverka):
    assert len(sverka) == 18 * 4 and set(sverka["status"]) == {"совпало"}
    assert (sverka["etalon"] == "da").all() and (sverka["bar_poz_razl"] == 0).all() and sverka["max_dpnl"].max() <= ek.PNL_TOL
    assert sorted(set(sverka["god"])) == [2018, 2019] and np.allclose(sorted(set(sverka["comm_pct"])), [0.035, 0.04], atol=1e-12)


@real
def test_mutatsiya_pozitsii_so_sdvigom_lovitsya(monkeypatch):
    from fastbt.sweep_sk import SkEvaluator
    orig = sweep_post.positions
    monkeypatch.setattr(sweep_post, "positions", lambda *a, **k: np.roll(orig(*a, **k), 1))
    r = ek.run_sk(KAND[0], SkEvaluator("S1"), (0.0004,))
    assert any(x["status"] == "РАСХОЖДЕНИЕ" for x in r)


# ---------- сухая сверка дат (риск вердикта #335: ValueError под замком) ----------

def daty_ok(cand: list, baza: list) -> bool:
    return cand == baza


def test_mutatsiya_dat_lovitsya():
    d = ["2018-01-09", "2018-01-10", "2019-12-30"]
    assert daty_ok(d, list(d)) and not daty_ok(d, d[1:]) and not daty_ok(d, d[:-1])


@real
def test_daty_kandidatov_i_s9baza_na_train_sovpadayut():
    from fastbt.sweep_a import SkPanelSource, make_sweep
    from fastbt.sweep_val_baza import S9Baza
    sp, bazy, _ = kd.build_spisok(".")
    assert len(sp) == 24
    bz, srcs = {}, {}
    for m in sp:
        ow = bz.setdefault(m["baza"]["id"], S9Baza(m["baza"]))
        rb = ow.evaluate({}, "train", 0.04)
        key = repr(sorted(m["tochki"].items(), key=str))
        sw = make_sweep(srcs.setdefault(key, SkPanelSource(m["tochki"])))
        rc = sw.evaluate(m["params"], "train", 0.04)
        assert daty_ok(list(rc["daty"]), list(rb["daty"])), m["id"]
        assert len(rc["daty"]) == len(rc["excess_d"]) == len(rb["excess_d"]) > 400, m["id"]


# ---------- реестр ----------

REE = pd.read_csv("docs/otbor/reestr_otbora_A.csv", comment="#")


def test_reestr_dopisan_i_starye_stroki_na_meste():
    assert (REE["istochnik"] == "sweep_e1").sum() == 1170 and (REE["istochnik"] == "sweep_e1v2").sum() == 2453
    z = REE[REE["istochnik"] == pz.ZAM]
    kand = z[z["status"] == "kandidat_zamorozki"]
    top = KA[KA["top3"] == 1]
    assert sorted(zip(kand["strategiya"], kand["klyuch"])) == sorted(zip(top["strategiya"], top["klyuch"]))
    assert (z["status"].isin(["kandidat_zamorozki", "otbroshena_net_etalona", "otbroshena_rang"])).all()
    assert set(REE.loc[REE["status"] == "otbroshena_net_etalona", "ekviv_s9"]) == {"net"}


def test_reestr_byvshie_kandidaty_s3_s6_net_etalona():
    s = REE[(REE["istochnik"] == pz.ZAM) & (REE["strategiya"].isin(["S3", "S4", "S5", "S6"]))]
    old = REE[(REE["istochnik"] == "sweep_e1v2") & (REE["status"] == "kandidat_zamorozki") & REE["strategiya"].isin(["S3", "S4", "S5", "S6"])]
    assert len(old) == 12
    bez = old[[ef.prichina_net(r.strategiya, pz.tochka(r.strategiya, r.klyuch)) is not None for r in old.itertuples()]]
    assert len(bez) > 0                                             # √RV/маски/trail_n среди бывших кандидатов есть
    got = s[s["klyuch"].isin(bez["klyuch"]) & (s["status"] != "kandidat_zamorozki")]
    assert (got["status"] == "otbroshena_net_etalona").all() and len(got) == len(bez)
    assert got["prichina"].str.contains("бывший кандидат").all()
    rest = s[s["klyuch"].isin(old["klyuch"]) & ~s["klyuch"].isin(bez["klyuch"])]
    assert set(rest["status"]) <= {"otbroshena_rang", "kandidat_zamorozki"}


def test_mutatsiya_reestr_bez_filtra_lovitsya(monkeypatch):
    base = kd.sk.expand(Path("fastbt/designs/sweep/S4_setka_e1b.yaml"))["base"]
    row = {"strategiya": "S4", "istochnik": "sweep_e1v2", "klyuch": "rm_edinitsa=sqrt_rv", "status": "otbroshena_rang"}
    assert base["rm_edinitsa"] == "atr14"
    ok = pz.stroki([row], {}, pd.DataFrame())
    assert [r["status"] for r in ok] == ["otbroshena_net_etalona"]
    monkeypatch.setattr(ef, "prichina_net", lambda sid, p: None)
    assert pz.stroki([row], {}, pd.DataFrame()) == []
