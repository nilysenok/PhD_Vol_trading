"""Свод wf (`fastbt.svod_wf`) на синтетическом каталоге возврата сервера: ledger 2.1 (v5) и 2.2 (v6), period = wf, проходы s1/s2/k3/spravka.
Счёта и данных нет: ряды случайные, режим известен по построению (до 2018 пик Шарпа при a = 2, в 2019 — резкий пик при a = 8).
Мутации: № 8/21 окно с годом t (K21.segment_okna → +4) — test_okno_ne_soderzhit_god_t; № 14 N = 1 в DSR — test_dsr_n_iz_obedineniya;
spravka в N — test_spravka_ne_v_n; к2 со сдвигом 0 ≠ исходный — test_k2_sdvig_nol; № 42 правило PBO подменено — test_pbo_pravilo_procedury."""
import csv
import itertools
import math

import numpy as np
import pytest

from fastbt import kalibr_v5_21 as K21, ledger_v2 as lm, metriki_e as ME, pbo_wf as PW, svod_wf as W, svod_wf_raschet as R

DATY = np.array([d for d in np.arange("2015-01-01", "2026-03-01", dtype="datetime64[D]") if np.is_busday(d)])
GOD = DATY.astype("datetime64[Y]").astype(int) + 1970
OSI = {"a": list(range(10)), "b": list(range(5))}
OSI_S = {"S1": {k: [lm.tok(v) for v in vs] for k, vs in OSI.items()}}
TOCHKI = [dict(zip(OSI, v)) for v in itertools.product(*OSI.values())]
OBEZ_TIK = np.ones((lm.NTIK, DATY.size))


def mu(pt, god):
    b = 0.13 * (pt["b"] - 2) ** 2
    return 40.0 - 4.0 * (pt["a"] - 8) ** 2 - b if god == 2019 else 2.0 - 0.15 * (pt["a"] - 2) ** 2 - b


def ryad(pt, sd=0.01):
    z = np.random.default_rng(pt["a"] * 10 + pt["b"]).standard_normal(DATY.size)
    return np.array([sd * (z[i] + mu(pt, g) / math.sqrt(252.0)) for i, g in enumerate(GOD)])


def zapisat(root, setka, tochki, rol="sobol", klyuch="seed", k3=False, kod="k", extra=()):
    v22 = setka == "v6"
    man = {"T": DATY.size, "kod": kod, "vhody": "v", "period": "wf", "k3": k3, "sobol_klyuch": klyuch, "komissiya": 0.0004, **({"setka": "v6"} if v22 else {})}
    w = (lm.Pisatel22 if v22 else lm.Pisatel21)(root, "S1", 0, man, shard=64, daty=DATY, poz=True)
    poz = (OBEZ_TIK[0], OBEZ_TIK, np.zeros(DATY.size), np.zeros_like(OBEZ_TIK))
    for i, (pt, r_) in enumerate([(p, rol) for p in tochki] + list(extra)):
        kl = lm.klyuch("S1", pt, "wf", 0.0004, None)
        row = {"status": "ok", "sobol_i": i, "v": lm.V22 if v22 else lm.V21, "strategiya": "S1", "klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, kod, "v"),
               "kod": kod, "vhody": "v", "by_year": {}, "by_quarter": {}}
        if v22:
            row.update(setka="v6", rol=r_, klyuch_n=lm.klyuch_n("S1", kl))
        x = ryad(pt)
        w.add(row, x, np.tile(x, (lm.NTIK, 1)), poz)
    w.close()


@pytest.fixture(scope="module")
def katalog(tmp_path_factory):
    root = tmp_path_factory.mktemp("wf")
    v5 = [p for p in TOCHKI if p["b"] < 4]
    chet = [p for p in TOCHKI if (p["a"] + p["b"]) % 2 == 0]
    dop = [dict(a=10, b=0), dict(a=10, b=1)]                       # точки справки: вне сетки, в N не входят
    zapisat(root / "v5_s1", "v5", v5, kod="k5")
    zapisat(root / "v6_s1", "v6", TOCHKI, kod="k6", extra=[(dict(a=3, b=3), "kontrol")])
    zapisat(root / "v6_s2", "v6", chet, klyuch="seed2", kod="k6")
    zapisat(root / "v6_k3", "v6", [p for p in TOCHKI if p["b"] == 2], k3=True, kod="k6")
    zapisat(root / "v6_spravka", "v6", dop, klyuch="spravka", rol="faza3", kod="k6")
    return root


def chitat(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    return [x for x in lines if x.startswith("#")], list(csv.DictReader([x for x in lines if not x.startswith("#")], delimiter=";"))


@pytest.fixture(scope="module")
def tablicy(katalog, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    paths = W.svod(katalog, out, OSI_S)
    return {p.stem: chitat(p) for p in paths}


def pul_v6(katalog):
    return R.pul_wf(W.prohody(katalog)["S1"]["v6", "s1"]["ch"], OSI_S["S1"])


def test_prohody_opredelyayutsya_po_manifest(katalog):
    ps = W.prohody(katalog)["S1"]
    assert set(ps) == {("v5", "s1"), ("v6", "s1"), ("v6", "s2"), ("v6", "k3"), ("v6", "spravka")}


def sredn_a(pul, w, t):
    return np.mean([float(pul.tochki[i]["a"]) for i in w["vybor"][t]["sel"][0]])


def test_okno_ne_soderzhit_god_t(katalog, monkeypatch):
    pul = pul_v6(katalog)
    w = R.wf(pul)
    assert len(w["vybor"]) == 10 and w["vybor"][2019]["okno"][1] == "2018-12-31" and w["vybor"][2026]["okno"][1] == "2025-12-31"
    a19 = sredn_a(pul, w, 2019)
    assert a19 < 4 and sredn_a(pul, w, 2020) > a19 + 2                   # окно 2019 не видит 2019; окно 2020 уже видит
    monkeypatch.setattr(K21, "segment_okna", lambda t, g0, n: (t - g0) * 4 + 3)       # мутация: окно захватывает год t
    pul2 = pul_v6(katalog)
    assert sredn_a(pul2, R.wf(pul2), 2019) > a19 + 2                     # выбор 2019 «знает» 2019: проверка выше на мутации падает


def test_k2_sdvig_nol(katalog, monkeypatch):
    pul = pul_v6(katalog)
    w = R.wf(pul)
    k2 = R.k2(pul, w["vybor"])
    assert k2["sharpe"] == pytest.approx(w["sharpe"], abs=1e-12) and len(k2["perm"]) == 9
    assert 0.1 <= k2["p"] <= 1.0
    assert R.k1(pul)["sharpe"] == pytest.approx(R.itog({t: R.summy_goda(pul, t, [(i, k) for i in range(len(pul.tochki)) for k in range(17)]) for t in R.GODY})["sharpe"])


def test_k1_pustoj_prefiks_otkaz(katalog):
    pul = pul_v6(katalog)
    with pytest.raises(ValueError):
        R.k1(pul, n=0)


def test_spravka_ne_v_n(tablicy, katalog, monkeypatch):
    _, rows = tablicy["wf_dsr_pbo"]
    assert int(rows[0]["N"]) == len(TOCHKI) == int(rows[0]["N_osnovnaya"])      # v5 ∪ v6 = 50 точек сетки; kontrol (3,3) и spravka (a = 10) не в N
    orig = W.rol_prohoda
    monkeypatch.setattr(W, "rol_prohoda", lambda mf: "s2" if mf.get("sobol_klyuch") == "spravka" else orig(mf))
    ob = lm.Obedinenie(*[x["ch"] for x in W.prohody(katalog)["S1"].values()])
    assert ob.n_k() == len(TOCHKI) + 2                    # мутация: spravka в объединении → N растёт, значит тест выше падает


def test_dsr_n_iz_obedineniya(tablicy, katalog, monkeypatch):
    r = tablicy["wf_dsr_pbo"][1][0]
    assert 0.0 <= float(r["dsr"]) <= float(r["dsr_N1_spravochno"]) <= 1.0
    ps = W.prohody(katalog)["S1"]
    x = np.array([ryad(p) for p in TOCHKI])                       # известный ответ: дневной Шарп точек из рядов, не из сумм
    sr = x.mean(1) / x.std(1, ddof=1)
    assert float(r["V_dnevnoj"]) == pytest.approx(float(np.var(sr, ddof=1)), abs=1e-6)
    assert float(r["sr_zvezda"]) == pytest.approx(ME.sr_zvezda(50, float(np.var(sr, ddof=1))), abs=1e-5)
    ob = lm.Obedinenie(*[ps[k]["ch"] for k in (("v5", "s1"), ("v6", "s1"), ("v6", "s2"), ("v6", "k3"))])
    nb = np.asarray(ps["v6", "s1"]["ch"].manifest["nbars"], float)
    d = R.dsr_wf(ob, nb, 4.0, 300)                                 # Шарп 4 (годовой) на 300 днях: N = 47…50 штрафует, N = 1 — нет
    assert d["N"] == ob.n_k() == 50 and 0.0 < d["dsr"] < d["dsr_n1"] <= 1.0
    monkeypatch.setattr(ob, "n_k", lambda: 1)                      # мутация № 14: N = 1
    m = R.dsr_wf(ob, nb, 4.0, 300)
    assert m["N"] == 1 and m["dsr"] > d["dsr"]                      # на мутации DSR вырос, N == 50 выше падает


def test_pbo_pravilo_procedury(katalog, monkeypatch):
    ps = W.prohody(katalog)["S1"]
    chs = [ps[k]["ch"] for k in (("v5", "s1"), ("v6", "s1"))]
    ob = lm.Obedinenie(*chs)
    zovy = []
    orig = PW.pravilo_procedury
    monkeypatch.setattr(R.PW, "pravilo_procedury", lambda sosedi, **kw: zovy.append(sosedi) or orig(sosedi, **kw))
    r = R.pbo_ob(ob, chs, OSI_S["S1"], "S1")
    assert len(zovy) == 1 and len(zovy[0]) == len(TOCHKI) and any(zovy[0])      # правило процедуры (не argmax) и соседи построены
    assert 0.0 <= r["pbo"] <= 1.0 and r["n"] + r["n_propusk"] == 252


def test_tablicy_shapka_i_formy(tablicy):
    for name, (shapka, rows) in tablicy.items():
        assert shapka[0] == "# статус: после аудита / тест-wf" and shapka[1].startswith("# код: ") and shapka[2].startswith("# входы: ")
        assert shapka[3].startswith("# предрегистрация v2 sha256: ") and rows, name
    okna = tablicy["wf_okna"][1]
    assert {r["god"] for r in okna} == {str(t) for t in range(2017, 2027)} and {r["setka"] for r in okna} == {"v5", "v6"}
    itog = {(r["setka"], r["prohod"]): r for r in tablicy["wf_itog"][1]}
    assert set(itog) == {("v5", "s1"), ("v6", "s1"), ("v6", "s2"), ("v6", "k3")}     # spravka не считается как проход
    assert len(tablicy["wf_seed2"][1]) == 10 and {r["kontrol"] for r in tablicy["wf_kontroli"][1]} == {"k1", "k2", "odna_tochka"}
