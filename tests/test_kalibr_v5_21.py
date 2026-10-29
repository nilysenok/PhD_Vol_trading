"""К-1: пул калибратора на ledger 2.1 (`fastbt/kalibr_v5_21.py`) на синтетике с известным ответом (mutations.md № 6, 8, 21).
Эталон — пул 2.0 (`kalibr_v5.Pul`) на тех же рядах: ключи членов, порядок и Шарп совпадают. Данных и сети нет."""
import itertools
import json

import numpy as np
import pytest

from fastbt import kalibr_v5 as K, kalibr_v5_21 as K21, ledger_v2 as lm, stats

DATY = np.array([d for d in np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]") if np.is_busday(d)])
OSI = {"a": list(range(10)), "b": list(range(5))}
LUCH = {"a": 6, "b": 2}
NK, KOM = 3, 0.002


def konec(t):
    return int((DATY < np.datetime64(f"{t}-01-01")).sum())


def ryad(rng, sh, n, sd=0.01):
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return sd * (z + sh / np.sqrt(252.0))


def dannye(spike=None, seed=1):
    """(tochki, ex[P, NK, T], poz[P, NK, T]): гладкая чаша Шарпа с вершиной LUCH; позиция открыта на конце каждого года."""
    rng = np.random.default_rng(seed)
    tochki = [dict(zip(OSI, v)) for v in itertools.product(*OSI.values())]
    ex = np.zeros((len(tochki), NK, DATY.size))
    for i, pt in enumerate(tochki):
        sh = 2.0 - 0.1 * ((pt["a"] - 6) ** 2 + (pt["b"] - 2) ** 2)
        for k in range(NK):
            ex[i, k] = ryad(rng, sh + 0.3 * k, DATY.size)
    poz = rng.choice([-1.0, 0.0, 1.0], size=ex.shape)
    for t in (2017, 2018, 2019):
        poz[..., konec(t) - 1] = 1.0
    if spike is not None:
        ex[tochki.index(LUCH), :, konec(2018)] = spike
    return tochki, ex, poz


def pula(tochki, ex, poz, kom=KOM, dos=None, dos21=None):
    p20 = K.Pul("S1", DATY, OSI, tochki, ex, poz, kom, dos)
    sm = lm.summy(DATY, ex, np.abs(poz))
    p21 = K21.Pul21("S1", OSI, tochki, sm, lm.nbars_kvartalov(DATY), 2016, kom, dos21)
    return p20, p21


def sravnit(r20, r21):
    for k, x in r20["tikery"].items():
        y = r21["tikery"][k]
        assert (x is None) == (y is None)
        if x:
            assert x["chleny"] == y["chleny"]
            assert np.allclose(x["plato"], y["plato"], rtol=0, atol=1e-12)
    assert r20["n_proby"] == r21["n_proby"]


def test_raven_pulu_20_sharp_top50_plato():
    tochki, ex, poz = dannye()
    p20, p21 = pula(tochki, ex, poz)
    for t in (2017, 2018, 2019):
        s, e = 0, K.granica(DATY, t)
        sh20 = K.metriki(ex[:, :, s:e].reshape(-1, e), poz[:, :, s:e].reshape(-1, e), KOM)[0].reshape(len(tochki), NK)
        assert np.abs(p21.okno(t)[0] - sh20).max() <= 1e-12
        for k in range(NK):
            assert faza1_ravny(p20, p21, k, t)
        sravnit(K.god(p20, t, "2016-01-01"), K21.god(p21, t))


def faza1_ravny(p20, p21, k, t):
    s, e = 0, K.granica(DATY, t)
    sh0, top0 = K.faza1(p20, k, s, e)
    sh1, top1 = K21.faza1(p21, k, t)
    return top0 == top1 and np.abs(sh0 - sh1).max() <= 1e-12


def test_raven_pulu_20_dosschitannye_sosedi():
    tochki, ex, poz = dannye()
    drop = lambda pt: pt["a"] == 6 and pt["b"] in (1, 3)           # у вершины нет двух соседей → досчёт
    keep = [i for i, pt in enumerate(tochki) if not drop(pt)]
    full = {K.klyuch("S1", pt): i for i, pt in enumerate(tochki)}
    sm = lm.summy(DATY, ex, np.abs(poz))
    ids = lambda q: full[K.klyuch("S1", q)]
    t2 = [tochki[i] for i in keep]
    p20, p21 = pula(t2, ex[keep], poz[keep], dos=lambda q: (ex[ids(q)], poz[ids(q)]), dos21=lambda q: sm[ids(q)])
    r20, r21 = K.god(p20, 2018, "2016-01-01"), K21.god(p21, 2018)
    sravnit(r20, r21)
    assert r21["n_proby"] > len(t2)


def test_klyuch_perioda_v2_i_start_2015():
    assert "v2" in K.PERIODY and "V2" not in K.PERIODY and K.PERIODY["v2"][0] == "2015-01-01"
    nbars = np.full(20, 63.0)
    sm = np.random.default_rng(0).normal(0, 0.01, (6, 2, 20, stats.NF))
    p = K21.Pul21("S1", {"a": [0, 1, 2]}, [{"a": v} for v in (0, 1, 2) * 2], sm, nbars, 2015)
    r = K21.kalibrovat(p, "v2")
    assert sorted(r) == [2017, 2018, 2019] and r[2017]["okno"] == ("2015-01-01", "2016-12-31")
    with pytest.raises(ValueError):
        K21.kalibrovat(p, "train4")


def lider(p21, t=2018):
    return K21.god(p21, t)["tikery"][0]["chleny"][0]


def test_mutaciya_okno_s_godom_t_lovitsya(monkeypatch):
    """Спайк −50 % на первом баре 2018 у лидера окна: окно до Q4 2017 его не видит, сдвиг на j+1 (Q1 2018) или j+4 (весь 2018) — видит."""
    tochki, ex, poz = dannye()
    lid = tochki[K21.faza1(pula(tochki, ex, poz)[1], 0, 2018)[1][0]]        # лидер окна 2016–2017 без спайка
    ex[tochki.index(lid), :, konec(2018)] = -0.5
    _, p21 = pula(tochki, ex, poz)
    assert lider(p21) == K.klyuch("S1", lid)
    j = K21.segment_okna
    for sdvig in (1, 4):
        monkeypatch.setattr(K21, "segment_okna", lambda t, g, n, s=sdvig: min(j(t, g, n) + s, n - 1))
        _, p = pula(tochki, ex, poz)
        assert lider(p) != K.klyuch("S1", lid)


def test_mutaciya_vykhod_na_konce_okna_ne_oplachen_lovitsya():
    tochki, ex, poz = dannye()
    p20, _ = pula(tochki, ex, poz)
    _, p0 = pula(tochki, ex, poz, kom=0.0)                          # пул с комиссией 0 вместо 0,002
    _, p21 = pula(tochki, ex, poz)
    sh20 = K.metriki(ex[:, 0, :K.granica(DATY, 2018)], poz[:, 0, :K.granica(DATY, 2018)], KOM)[0]
    assert np.abs(p21.okno(2018)[0][:, 0] - sh20).max() <= 1e-12
    assert np.abs(p0.okno(2018)[0][:, 0] - sh20).max() > 1e-6


def test_otkaz_okno_2016_ledger_20_bez_poz_konec(tmp_path):
    tochki, ex, poz = dannye()
    _, p21 = pula(tochki, ex, poz)
    with pytest.raises(ValueError):
        K21.segment_okna(2016, 2016, 16)
    with pytest.raises(ValueError):
        K21.segment_okna(2021, 2016, 16)
    d = tmp_path / "S1"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"v": "2.0", "kod": "K", "vhody": "V", "T": 3}))
    (d / "p00").mkdir()
    with pytest.raises(ValueError):
        K21.pul_iz_ledger(tmp_path, "S1", OSI)


def test_pul_iz_ledger_21_sovpadaet_s_pulom(tmp_path):
    tochki, ex, poz = dannye()
    w = lm.Pisatel21(tmp_path, "S1", 0, {"T": DATY.size, "kod": "K", "vhody": "V"}, shard=64, daty=DATY, poz=True)
    for i, pt in enumerate(tochki[:12]):
        kl = lm.klyuch("S1", pt, "train4", KOM, None)
        row = {"klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, "K", "V"), "v": "2.1", "status": "ok", "kod": "K", "vhody": "V"}
        z = np.zeros((17, DATY.size))
        sd = np.zeros((17, DATY.size))
        tk, pz = z.copy(), z.copy()
        tk[:NK], pz[:NK] = ex[i], np.abs(poz[i])
        w.add(row, tk[0], tk, (pz[0], pz, sd[0], sd))
    w.close()
    p = K21.pul_iz_ledger(tmp_path, "S1", OSI)
    assert p.sm.shape == (12, 17, 16, stats.NF) and p.komissiya == KOM and len(p.nbars) == 16
    _, ref = pula(tochki[:12], ex[:12], poz[:12])
    assert np.abs(p.okno(2018)[0][:, :NK] - ref.okno(2018)[0]).max() <= 1e-12
