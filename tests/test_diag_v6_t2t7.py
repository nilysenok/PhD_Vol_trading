"""Д-3 диагностики v6 train4: таблицы (2), (7), оговорки. Известный ответ на синтетике + мутации (сети нет, реальных данных нет)."""
import json
import math

import numpy as np
import pytest

from fastbt import diag_v6_pravila as P
from fastbt import diag_v6_t2t7 as T
from fastbt import ledger_v2 as L

GOD = ("2016", "2017", "2018", "2019")
OK = {"stop": 0.5, "take": 0.1, "trail": 0.1, "bu": 0.0, "vremya": 0.0, "signal": 0.1, "part": 0.2}    # Σ словаря 0,8 → прочее 0,2
OK2 = {"stop": 0.2, "take": 0.1, "trail": 0.1, "bu": 0.0, "vremya": 0.0, "signal": 0.1, "part": 0.5}   # прочее 0,5


def _r(i, prohod, msr, sharpe, sd, pr, status="ok", sloy="fast", **fl):
    return {"klyuch_n": f"k{i}", "strategiya": "S1", "status": status, "rol": "sobol", "msr": msr, "sharpe": sharpe,
            "sdelok_y": {g: sd for g in GOD} if status == "ok" else {}, "prichiny": pr, "sloy": sloy, "sobol_i": i,
            "prohod": prohod, "brent": fl.get("brent", False), "stop_reentry": fl.get("sr", False)}


def _pul():
    """S1: ok-точки i=1…12 (msr = i, sharpe = 13 − i, сделок в год: i=1,2 → 2, остальные → i), 2 нулевых ok (13, 14), 4 ne_schitana."""
    rows = []
    for i in range(1, 13):
        rows.append(_r(i, "train4_v6_s1" if i <= 7 else "train4_v6_s2", float(i), 13.0 - i, 2 if i < 3 else i, OK if i <= 7 else OK2,
                       sloy="slow" if i == 3 else "fast", brent=i in (1, 4), sr=i == 5))
    rows += [_r(i, "train4_v6_s2", NAN, NAN, 0, {}, sloy="slow" if i == 14 else "fast", sr=i == 14) for i in (13, 14)]
    rows += [_r(20 + j, "train4_v6_s1", NAN, NAN, 0, {}, status="ne_schitana") for j in range(4)]
    for r in rows:
        r["prohod"] = "s1" if r["prohod"].endswith("s1") else "s2"
    return {"S1": rows}, {"S1": {"gody": [2016, 2017, 2018, 2019], "N": {"s1": 15, "s2": 7},
                                 "chitano": {"s1": 11, "s2": 7}}}


NAN = float("nan")
_TOP = T.top10_po_sk


@pytest.fixture(autouse=True)
def _bez_vorot(request, monkeypatch):
    """Синтетика G3.1–G3.4 — без ворот (min_sdelok=0); ворота проверяют тесты с меткой `vorota`."""
    if "vorota" not in request.node.name:
        monkeypatch.setattr(T, "top10_po_sk", lambda pul, m=0: _TOP(pul, m))


def _tochka(i, sdelok_god, **kw):
    return dict(_r(i, "s1", float(i), 1.0, sdelok_god, OK), **kw)


def test_vorota_29_ne_vhodit_30_vhodit():
    """Сделок за окно = Σ sdelok_y (4 года): 29 → нет, 30 → да; i=1…3 дают 7,25·4 = 29 / 7,5·4 = 30."""
    pul = {"S1": [_tochka(1, 7.25), _tochka(2, 7.5), _tochka(3, 20)]}
    assert [r["sobol_i"] for r in T.top10_po_sk(pul)["S1"]] == [3, 2]
    assert [r["sobol_i"] for r in T.top10_po_sk(pul, 0)["S1"]] == [3, 2, 1]
    assert [r["sobol_i"] for r in P.top10(pul["S1"], min_sdelok=30)] == [3, 2]


def test_mutaciya_vorota_po_prinyato_lovitsya(monkeypatch):
    """№ 41: ворота по `prinyato` (принятые сигналы) вместо реальных сделок: точка с 29 сделками, но большим prinyato, прошла бы."""
    pul = {"S1": [_tochka(1, 7.25, prinyato=500), _tochka(2, 7.5, prinyato=1)]}
    monkeypatch.setattr(P, "sdelok_okno", lambda r: float(r.get("prinyato", 0)))
    assert [r["sobol_i"] for r in T.top10_po_sk(pul)["S1"]] != [2]       # мутант пускает i=1 и не пускает i=2


def _vse(t2):
    return {r["prohod"]: r for r in t2}


def test_g31_tablica_2_izvestnyj_otvet():
    pul, meta = _pul()
    t = _vse(T.tablica_2(pul, meta))
    v = t["vse"]
    assert (v["n_strok"], v["n_ok"], v["n_ne_schitana"], v["n_oshibka"], v["n_nul"]) == (18, 14, 4, 0, 2)
    assert v["nuli_dolya"] == pytest.approx(2 / 14)              # нули — только среди ok
    assert v["vybrosheno"] == 4 and t["s1"]["vybrosheno"] == 4 and t["s2"]["vybrosheno"] == 0
    assert v["sloy_fast_dolya"] == pytest.approx(16 / 18)
    assert v["top10_n"] == 10 and v["sdelok_v_god_mediana"] == 7.5 and v["sdelok_2018"] == 7.5   # топ-10 = i 3…12 по msr
    assert v["slabye"] == "sdelok_malo"                           # 7,5 < 10; нулей 1/7 ≤ 50 %
    assert t["s2"]["nuli_dolya"] == pytest.approx(2 / 7)
    assert "slabye" not in t["s1"]
    assert "слабое место: sdelok_malo, Sk: S1" in T.vyvod(T.tablica_2(pul, meta), "(2)")


def test_g31_ne_slaboe_mesto():
    pul, meta = _pul()
    for r in pul["S1"]:
        r["sdelok_y"] = {g: 50 for g in GOD} if r["status"] == "ok" and r["msr"] == r["msr"] else r["sdelok_y"]
    assert T.vyvod(T.tablica_2(pul, meta), "(2)").startswith("(2): не слабое место")


def test_g32_mutaciya_top10_po_drugomu_polyu_lovitsya(monkeypatch):
    pul, meta = _pul()
    monkeypatch.setitem(P.TOP10, "kluch", "sharpe")               # топ-10 = i 1…10: сделок 2, 2, 3…10 → медиана 5,5 ≠ 7,5
    v = _vse(T.tablica_2(pul, meta))["vse"]
    assert v["sdelok_v_god_mediana"] == 5.5
    assert T.tablica_7(pul)[0]["sobol_i"] != 12                   # первым в топ-10 стоит не лучший по msr


def test_g33_tablica_7_prochee_izvestnyj_otvet():
    pul, _ = _pul()
    t = T.tablica_7(pul)
    assert [r["sobol_i"] for r in t if r["rang"] != "med"] == list(range(12, 2, -1))
    med = t[-1]
    assert med["rang"] == "med" and med["prochee"] == pytest.approx(0.35) and med["stop"] == pytest.approx(0.35)
    assert med["slabye"] == "prochee_mnogo"                       # 0,35 > 0,3
    assert t[0]["prochee"] == pytest.approx(0.5) and t[-2]["prochee"] == pytest.approx(0.2)
    assert "prochee_mnogo, Sk: S1" in T.vyvod(t, "(7)")


def test_g33_mutaciya_prochee_otbrosheno_lovitsya(monkeypatch):
    pul, _ = _pul()
    monkeypatch.setattr(T, "_prochee", lambda r: 0.0)
    med = T.tablica_7(pul)[-1]
    assert med["prochee"] != pytest.approx(0.35) and med["slabye"] == ""


def test_prochee_bez_sdelok_nan_i_summa_bolshe_edinicy_oshibka():
    assert math.isnan(T._prochee({"sdelok_y": {"2016": 0}, "prichiny": {"stop": 1}}))
    with pytest.raises(ValueError):
        T._prochee({"sdelok_y": {"2016": 3}, "prichiny": {"stop": 0.9, "take": 0.3}})


def test_g34_mutaciya_nuli_po_vsem_statusam_lovitsya(monkeypatch):
    pul, meta = _pul()
    monkeypatch.setattr(T, "_nul", lambda r: T._summa(r) == 0.0)  # ne_schitana (сделок нет) станут «нулями»: 6 из 14 ok — бессмыслица
    v = _vse(T.tablica_2(pul, meta))["vse"]
    assert v["n_nul"] == 6 and v["nuli_dolya"] != pytest.approx(2 / 14)


def test_ogovorki_izvestnyj_otvet():
    pul, meta = _pul()
    o = {(r["nazvanie"]): r for r in T.ogovorki(pul, meta)}
    assert (o["yadro_sloy_fast"]["chislo"], o["yadro_sloy_fast"]["vsego"]) == (16, 18)
    assert o["brent_dni_v_okne"]["chislo"] == 0 and o["brent_dni_v_okne"]["vsego"] == 38 and o["brent_dni_v_okne"]["tekst"] == "не задевает"
    assert (o["brent_tochki"]["chislo"], o["brent_tochki"]["vsego"]) == (1, 2)             # i=4 в топ-10; i=1 только в пуле
    sr = o["stop_reentry_tochki"]
    assert (sr["chislo"], sr["vsego"]) == (1, 2) and "slow: топ-10 0, пул 1" in sr["tekst"]


def test_brent_zadevaet_okno_s_2021():
    assert T.dni_brent_v_okne([2016, 2017, 2018, 2019]) == 0
    assert T.dni_brent_v_okne([2020, 2021]) == 38
    pul, meta = _pul()
    meta["S1"]["gody"] = [2021]
    assert [r for r in T.ogovorki(pul, meta) if r["nazvanie"] == "brent_dni_v_okne"][0]["tekst"] == "ЗАДЕВАЕТ окно"


def test_szhat_flagi_iz_klyucha():
    def f(sid, **k):
        return T.szhat({"strategiya": sid, "klyuch": k, "klyuch_n": "x"}, "s1")
    assert f("S1", vn_uslovie="po_bete")["brent"] and not f("S1", vn_uslovie="po_bete", vn_faktor="rvi")["brent"]
    assert not f("S1", vn_uslovie="net")["brent"] and not f("S1")["brent"]
    assert f("S3", reentry="da")["stop_reentry"] and not f("S3", reentry="net")["stop_reentry"]
    assert not f("S3", reentry="da", vhod_tip="podtverzhdenie")["stop_reentry"] and not f("S1", reentry="da")["stop_reentry"]


DATY = np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]")


def _pisat(root, idx, N):
    man = {"T": len(DATY), "kod": "k", "vhody": "v", "setka": "s", "period": "train4"}
    w = L.Pisatel22(root, "S1", 0, man, shard=4, daty=DATY)
    for i in idx:
        kl = L.klyuch("S1", {"a": i}, "train4", 0.0004, None)
        rol = "kontrol" if i == 99 else "sobol"
        row = {"v": "2.2", "strategiya": "S1", "klyuch": kl, "klyuch_n": L.klyuch_n("S1", kl), "klyuch_sha": L.klyuch_sha(kl, "k", "v"),
               "kod": "k", "vhody": "v", "setka": "s", "rol": rol, "status": "ok", "sharpe": float(i), "msr": float(i), "sloy": "fast",
               "sobol_i": i, "sdelok_y": {g: 2.0 for g in GOD}, "prichiny": {"stop": 1.0}}
        rng = np.random.default_rng(i)
        w.add(row, rng.normal(0, 1e-3, len(DATY)), rng.normal(0, 1e-3, (17, len(DATY))))
    w.close()
    (root / "S1" / "checkpoint.json").write_text(json.dumps({"N": N}))


def test_chitat_pul_s1_s2_dubl_i_kontrol(tmp_path):
    _pisat(tmp_path / "train4_v6_s1", [0, 1, 2, 3, 99], 10)
    _pisat(tmp_path / "train4_v6_s2", [3, 4], 5)                  # i=3 — тот же klyuch_n в обеих выборках
    pul, meta = T.chitat_pul(tmp_path)
    assert sorted(r["sobol_i"] for r in pul["S1"]) == [0, 1, 2, 3, 4]       # rol kontrol не в пуле, дубль один раз
    assert meta["S1"]["N"] == {"s1": 10, "s2": 5} and meta["S1"]["gody"] == [2016, 2017, 2018, 2019]
    v = _vse(T.tablica_2(pul, meta))["vse"]
    assert v["n_strok"] == 5 and v["vybrosheno"] == 15 - 6        # N 15, прочитано 4 sobol (s1) + 2 (s2) = 6
    assert [r["sobol_i"] for r in T.tablica_7(pul) if r["rang"] != "med"] == [4, 3, 2, 1, 0]
    assert T.tablica_7(pul)[0]["prochee"] == 0.0
