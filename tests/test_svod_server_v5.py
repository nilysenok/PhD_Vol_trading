"""Б5: свод возврата сервера (`fastbt.svod_server_v5`) на малом синтетическом ledger (счёта Соболя и данных нет).
Мутации (mutations.md № 8/21, № 27): окно отбора до 31.12 t вместо 31.12(t−1) — G1/G3 падают; k1 по всем годам — G2 падает."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fastbt import kalibr_v5 as K, kontroli_v5 as kv, ledger_v2 as lm, svod_server_v5 as SV
from fastbt.data import year_bounds
from fastbt.metriki_e import sr_dnevnoy

DATY = np.array([d for d in np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]") if np.is_busday(d)])
OSI = {"a": list(range(10)), "b": list(range(5))}
OSI_S = {"S1": {a: [lm.tok(v) for v in vs] for a, vs in OSI.items()}}
T_ = 2018
NSH = 19


def ryad(rng, sh, n, sd=0.01):
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return sd * (z + sh / np.sqrt(252.0))


def okno(p):
    return 2.0 - 0.1 * (p["a"] - 6) ** 2 - 0.13 * (p["b"] - 2) ** 2 - 0.01 * p["a"]      # без точных ничьих


def god_t(p):
    return 6.0 - 0.5 * ((p["a"] - 1) ** 2 + (p["b"] - 1) ** 2)


def tochki():
    return [dict(zip(OSI, v)) for v in itertools.product(*OSI.values())]


def ryady(surf, surf_t, seed):
    rng, e, (s_t, e_t) = np.random.default_rng(seed), K.granica(DATY, T_), year_bounds(DATY, T_)
    out = []
    for pt in tochki():
        x = np.zeros(DATY.size)
        x[:e] = ryad(rng, surf(pt), e)
        x[s_t:e_t] = ryad(rng, surf_t(pt), e_t - s_t)
        out.append(x)
    return out


def zapisat(root, sid="S1", v21=False, seed=1, klyuch="seed", k3=False, surf=okno, surf_t=god_t):
    man = {"T": DATY.size, "kod": "k", "vhody": "v", "period": "train4", "k3": k3, "sobol_klyuch": klyuch, "komissiya": 0.0004}
    w = lm.Pisatel21(root, sid, 0, man, daty=DATY, poz=True) if v21 else lm.Pisatel(root, sid, 0, man)
    oni, nul = np.ones(DATY.size), np.zeros(DATY.size)
    poz = (oni, np.tile(oni, (lm.NTIK, 1)), nul, np.tile(nul, (lm.NTIK, 1)))      # |позиция| = 1: экспозиция 100 %
    for i, (pt, x) in enumerate(zip(tochki(), ryady(surf, surf_t, seed))):
        kl = lm.klyuch(sid, pt, "train4", 0.0004, None)
        row = {"status": "ok", "sobol_i": i, "v": lm.V21 if v21 else lm.V, "klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, "k", "v"),
               "kod": "k", "vhody": "v", "by_year": {}, "by_quarter": {}}
        w.add(row, x, np.tile(x, (lm.NTIK, 1)), *([poz] if v21 else []))
    w.close()
    np.save(Path(root) / sid / "daty.npy", DATY)


def chitat(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    shapka = [x for x in lines if x.startswith("#")]
    rows = list(csv.DictReader([x for x in lines if not x.startswith("#")], delimiter=";"))
    return shapka, rows


def top10(rows, god=T_, tiker="0"):
    return [r["tochka"] for r in rows if r["god"] == str(god) and r["tiker"] == tiker]


def ozhidaemoe():
    best = sorted(tochki(), key=lambda p: -okno(p))[:10]
    return {K.klyuch("S1", p) for p in best}


@pytest.fixture
def vozvrat(tmp_path):
    zapisat(tmp_path / "sobol1_seed1", seed=1)
    zapisat(tmp_path / "sobol1_seed2", seed=2, klyuch="seed2")
    zapisat(tmp_path / "k3", seed=3, k3=True, klyuch="seed")
    return tmp_path


def gotovo(vozvrat, tmp_path):
    SV.svod(vozvrat, tmp_path / "out", OSI_S, NSH)
    return tmp_path / "out"


def test_g1_izvestnyj_top10_vybran(vozvrat, tmp_path):
    out = gotovo(vozvrat, tmp_path)
    _, rows = chitat(out / "kalibr_top10.csv")
    pr = [r for r in rows if r["prohod"] == "S1|train4|v5|seed"]
    assert len(top10(pr)) == 10 and set(top10(pr)) == ozhidaemoe()
    assert {r["okno_po"] for r in pr if r["god"] == str(T_)} == {"2017-12-29"}        # ни одного бара года t


def test_g2_k1_k2_ravny_pryamomu_vyzovu(vozvrat, tmp_path):
    out = gotovo(vozvrat, tmp_path)
    _, rows = chitat(out / "kontroli_k1_k2_k3.csv")
    r = next(x for x in rows if x["prohod"] == "S1|train4|v5|seed")
    pul = K.pul_iz_ledger(vozvrat / "sobol1_seed1", "S1", OSI_S["S1"])
    k2 = kv.k2(pul, kv.kalibr(pul, kv.GODY, "2016-01-01"), NSH)
    ref1 = float(sr_dnevnoy(pul.ex[:, :, np.isin(kv._god(pul), kv.GODY)].mean((0, 1))) * np.sqrt(252))   # независимо от kv.k1
    assert abs(float(r["k1_sharpe"]) - kv.k1(pul)) <= 1e-6 and abs(kv.k1(pul) - ref1) <= 1e-12   # csv — 6 знаков
    assert abs(float(r["sharpe_ansambl_k2"]) - k2["sharpe"]) <= 1e-6 and abs(float(r["k2_p"]) - k2["p"]) <= 1e-6
    assert {x["prohod"] for x in rows} == {"S1|train4|v5|seed", "S1|train4|v5|seed2", "S1|train4|k3|seed"}


def test_g3_okno_do_31_12_t_minus_1_i_mutaciya(vozvrat, tmp_path, monkeypatch):
    def pervye():
        _, rows = chitat(gotovo(vozvrat, tmp_path) / "kalibr_top10.csv")
        return {r["tochka"] for r in rows if r["prohod"] == "S1|train4|v5|seed" and r["god"] == str(T_) and r["tiker"] == "0" and r["rang"] == "1"}
    assert pervye() <= ozhidaemoe()
    monkeypatch.setattr(K, "granica", lambda d, t: year_bounds(d, t)[1])           # мутация: окно до 31.12 t
    mut = pervye()
    assert not mut <= ozhidaemoe()


def test_g3_mutaciya_k1_po_vsem_godam_lovitsya(vozvrat, tmp_path, monkeypatch):
    pul = K.pul_iz_ledger(vozvrat / "sobol1_seed1", "S1", OSI_S["S1"])
    ok = kv.k1(pul)
    monkeypatch.setattr(kv, "GODY", (2016, 2017, 2018, 2019))
    assert kv.k1(pul, kv.GODY) != ok


def test_g4_ledger_21_chitaetsya_i_20_tozhe(tmp_path):
    zapisat(tmp_path / "v20")
    zapisat(tmp_path / "v21", v21=True)
    pr = {p["v"]: p for p in SV.prohody(tmp_path)}
    assert set(pr) == {lm.V, lm.V21}
    rows = SV.okna_v21(pr[lm.V21])
    r18 = [r for r in rows if r[1] == T_]
    assert len(r18) == 10 and {x[2] for x in r18} == set(range(1, 11))
    sh = [float(x[4]) for x in r18]
    assert sh == sorted(sh, reverse=True)
    lader = {tuple(sorted((k, v) for k, v in json.loads(x[3]).items() if k in ("a", "b"))) for x in r18}
    ref = {tuple(sorted((k, lm.tok(v)) for k, v in p.items())) for p in sorted(tochki(), key=lambda p: -okno(p))[:10]}
    assert lader == ref
    out = SV.svod(tmp_path, tmp_path / "out", OSI_S, NSH)                # оба ledger — одним вызовом
    _, okn = chitat(tmp_path / "out" / "okna_ledger21.csv")
    _, kal = chitat(tmp_path / "out" / "kalibr_top10.csv")
    assert len(okn) == 30 and {r["prohod"] for r in okn} == {"S1|train4|v5|seed"} and len(kal) == 3 * 17 * 10 and len(out) == 5


def test_g5_shapka_status_kod_hash(vozvrat, tmp_path):
    out = gotovo(vozvrat, tmp_path)
    for f in out.glob("*.csv"):
        sh, _ = chitat(f)
        assert sh[0] == f"# статус: {SV.STATUS}" and sh[1] == f"# код: {SV.kod_sha()}"
        assert sh[2].startswith("# входы: ") and len(sh[2].split()) == 2 + 3        # три прохода — три хэша
    sha0 = SV.prohody(vozvrat)[0]["vhod"]
    (vozvrat / "k3" / "S1" / "daty.npy").write_bytes(b"x")                          # подмена входа меняет хэш
    assert {p["vhod"] for p in SV.prohody(vozvrat)} != {sha0} and SV.vhod_sha(vozvrat / "k3", "S1") != sha0


def test_seed2_rashozhdenie_po_godam(vozvrat, tmp_path):
    _, rows = chitat(gotovo(vozvrat, tmp_path) / "seed2_ustojchivost.csv")
    assert [int(r["god"]) for r in rows] == [2017, 2018, 2019]
    p1 = K.pul_iz_ledger(vozvrat / "sobol1_seed1", "S1", OSI_S["S1"])
    p2 = K.pul_iz_ledger(vozvrat / "sobol1_seed2", "S1", OSI_S["S1"])
    ref = kv.seed2(SV.proc("train4"), p1, p2)
    for r in rows:
        assert abs(float(r["rashozhdenie"]) - ref[int(r["god"])]["rashozhdenie"]) <= 1e-6


def test_pustoj_vozvrat_dayot_pustye_tablicy(tmp_path):
    paths = SV.svod(tmp_path / "net", tmp_path / "out", OSI_S, NSH)
    assert len(paths) == 5 and all(len(chitat(p)[1]) == 0 for p in paths)
