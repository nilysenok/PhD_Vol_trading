"""Тестировщик #872: мутации Д-3 (ворота по prinyato, топ-10 только из s1, «прочее» отброшено, пометка по skew без модуля, дубль klyuch)."""
import json
import math

import numpy as np
import pytest

from fastbt import diag_v6_pravila as P
from fastbt import diag_v6_t2t7 as T
from fastbt import ledger_v2 as L

GOD = ("2016", "2017", "2018", "2019")
DATY = np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]")


def _row(i, sd, sharpe=None, pr=None, **kw):
    return dict({"klyuch_n": i, "status": "ok", "rol": "sobol", "sharpe": float(i) if sharpe is None else sharpe, "msr": float(i),
                 "sdelok_y": {g: sd for g in GOD}, "prichiny": pr or {"stop": 1.0}, "sloy": "fast", "sobol_i": i, "prohod": "s1"}, **kw)


def _pisat(root, idx, N):
    man = {"T": len(DATY), "kod": "k", "vhody": "v", "setka": "s", "period": "train4"}
    w = L.Pisatel22(root, "S1", 0, man, shard=4, daty=DATY)
    for i in idx:
        kl = L.klyuch("S1", {"a": i}, "train4", 0.0004, None)
        row = {"v": "2.2", "strategiya": "S1", "klyuch": kl, "klyuch_n": L.klyuch_n("S1", kl), "klyuch_sha": L.klyuch_sha(kl, "k", "v"),
               "kod": "k", "vhody": "v", "setka": "s", "rol": "sobol", "status": "ok", "sharpe": float(i), "msr": float(i), "sloy": "fast",
               "sobol_i": i, "sdelok_y": {g: 10.0 for g in GOD}, "prichiny": {"stop": 1.0}}
        rng = np.random.default_rng(i)
        w.add(row, rng.normal(0, 1e-3, len(DATY)), rng.normal(0, 1e-3, (17, len(DATY))))
    w.close()
    (root / "S1" / "checkpoint.json").write_text(json.dumps({"N": N}))


def test_vorota_summa_po_realnym_sdelkam_a_ne_po_prinyato():
    rows = [_row(1, 7.25, prinyato=900), _row(2, 7.5, prinyato=0), _row(3, 7.5, prinyato=0, status="ne_schitana")]
    assert [r["klyuch_n"] for r in P.top10(rows)] == [2]                    # 29 → нет, 30 → да; ne_schitana не входит
    assert P.sdelok_okno(_row(1, 7.25)) == 29.0 and P.sdelok_okno({"prinyato": 99}) == 0.0


def test_mutaciya_topy_tolko_iz_s1_lovitsya(tmp_path, monkeypatch):
    _pisat(tmp_path / "train4_v6_s1", [1, 2], 10)
    _pisat(tmp_path / "train4_v6_s2", [3, 4], 10)
    pul, _ = T.chitat_pul(tmp_path)
    assert {r["prohod"] for r in pul["S1"]} == {"s1", "s2"} and len(pul["S1"]) == 4
    assert len(T.top10_po_sk(pul)["S1"]) == 4 and T.top10_po_sk(pul)["S1"][0]["sobol_i"] == 4     # лучший — из s2
    monkeypatch.setattr(T, "PROHODY", {"s1": "train4_v6_s1"})                                       # мутация
    pul2, _ = T.chitat_pul(tmp_path)
    assert T.top10_po_sk(pul2)["S1"][0]["sobol_i"] != 4 and len(pul2["S1"]) == 2


def test_dubl_klyucha_odin_raz_mutaciya_bez_dedupa_lovitsya(tmp_path):
    _pisat(tmp_path / "train4_v6_s1", [1, 2], 5)
    _pisat(tmp_path / "train4_v6_s2", [2, 3], 5)
    pul, _ = T.chitat_pul(tmp_path)
    ks = [r["klyuch_n"] for r in T.top10_po_sk(pul)["S1"]]
    assert len(ks) == len(set(ks)) == 3
    bez_dedupa = [r for r in pul["S1"]] + [r for r in pul["S1"] if r["sobol_i"] == 2]
    ks2 = [r["klyuch_n"] for r in P.top10(bez_dedupa)]
    assert len(ks2) != len(set(ks2))                                        # мутант дал бы дубль в топ-10


def test_prochee_vse_vne_slovarya_summa_dolej_edinica(monkeypatch):
    r = _row(1, 5, pr={"stop": 0.6, "part": 0.2, "pir": 0.1, "sessiya": 0.1})      # вне словаря: 0,4
    assert T._prochee(r) == pytest.approx(0.4)
    pul = {"S1": [_row(i, 10, pr={**dict.fromkeys(T.SLOVAR, 0.0), "stop": 0.3, "take": 0.1, "part": 0.6}) for i in range(1, 4)]}
    t7 = [x for x in T.tablica_7(pul) if x["rang"] != "med"]
    assert len(t7) == 3                                                     # ворота пройдены (40 сделок), цикл не пустой
    for x in t7:
        assert sum(x[k] for k in T.SLOVAR) + x["prochee"] == pytest.approx(1.0)
    monkeypatch.setattr(T, "_prochee", lambda r: 0.0)                       # мутация: «прочее» отброшено
    x = [x for x in T.tablica_7(pul) if x["rang"] == 1][0]
    assert sum(x[k] for k in T.SLOVAR) + x["prochee"] != pytest.approx(1.0)


def test_mutaciya_msr_plohoy_bez_modulya_skew_lovitsya(monkeypatch):
    rows = [_row(1, 10, skew=-5.0, kurt=3.0), _row(2, 10, skew=0.1, kurt=3.0)]
    assert [r["msr_plohoy"] for r in P.top10(rows)] == [False, True]        # i=2 (sharpe 2) первый, i=1 с skew −5 помечен
    sk = lambda r: bool((r.get("skew") or 0) > P.SKEW_MAX or (r.get("kurt") or 0) > P.KURT_MAX)
    monkeypatch.setattr(P, "msr_plohoy", sk)
    assert [r["msr_plohoy"] for r in P.top10(rows)] != [False, True]
    assert not math.isnan(P.SKEW_MAX)


def test_nul_tolko_sredi_ok_ne_schitana_i_oshibka_ne_nuli():
    pul = {"S1": [_row(1, 0), _row(2, 5)] + [_row(i, 0, status=s) for i, s in ((3, "ne_schitana"), (4, "oshibka"))]}
    pul["S1"][2]["sdelok_y"], pul["S1"][3]["sdelok_y"] = {}, {}
    v = [r for r in T.tablica_2(pul, {"S1": {"N": {}, "chitano": {}}}) if r["prohod"] == "vse"][0]
    assert (v["n_nul"], v["n_ok"], v["n_ne_schitana"], v["n_oshibka"]) == (1, 2, 1, 1) and v["nuli_dolya"] == 0.5
