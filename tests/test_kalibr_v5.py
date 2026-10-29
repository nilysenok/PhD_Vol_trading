"""П7: калибратор v5 (`fastbt/kalibr_v5.py`) на синтетике с известным ответом и мутации (Ж-127 п. 3, mutations.md № 6, 27).
Детектор на правильном коде — True, на мутации (monkeypatch) — False. Данных и сети нет."""
import itertools
import json
import types

import numpy as np

from fastbt import kalibr_v5 as K, ledger_v2 as lm, sweep_plato as SP
from fastbt.approach_a import ApproachA
from fastbt.data import year_bounds
from fastbt.metriki_e import sr_dnevnoy

DATY = np.array([d for d in np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]") if np.is_busday(d)])
T, G = 2018, ("2016-01-01",)


def ryad(rng, sh: float, n: int, sd: float = 0.01) -> np.ndarray:
    """Ряд длины n с Шарпом ровно sh (годовая шкала, ddof = 1)."""
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return sd * (z + sh / np.sqrt(252.0))


def pul(surf, surf_t, osi, spike=None, poz=False, seed=1):
    """Пул 1 тикер: окно 2016–2017 с Шарпом surf(pt), год 2018 — surf_t(pt); spike = (pt, x) — первый бар 2018."""
    rng, e, (s_t, e_t) = np.random.default_rng(seed), K.granica(DATY, T), year_bounds(DATY, T)
    tochki = [dict(zip(osi, v)) for v in itertools.product(*osi.values())]
    ex = np.zeros((len(tochki), 1, DATY.size))
    for i, pt in enumerate(tochki):
        ex[i, 0, :e] = ryad(rng, surf(pt), e)
        ex[i, 0, s_t:e_t] = ryad(rng, surf_t(pt), e_t - s_t)
        if spike is not None and pt == spike[0]:
            ex[i, 0, s_t] = spike[1]
    pz = rng.choice([-1.0, 0.0, 1.0], size=ex.shape) if poz else None
    return K.Pul("S1", DATY, osi, tochki, ex, pz)


OSI = {"a": list(range(10)), "b": list(range(5))}
LUCH, LUCH_T = {"a": 6, "b": 2}, {"a": 1, "b": 1}


def leak_pul():
    """Лучший в окне — (6, 2); год 2018 лучший — (1, 1); у (6, 2) первый бар 2018 = −50 %."""
    return pul(lambda p: 2.0 - 0.1 * ((p["a"] - 6) ** 2 + (p["b"] - 2) ** 2),
               lambda p: 6.0 - 0.5 * ((p["a"] - 1) ** 2 + (p["b"] - 1) ** 2), OSI, spike=(LUCH, -0.5))


def pervyj(pul_, t=T):
    r = K.god(pul_, t, G[0])["tikery"][0]
    return r["chleny"][0] if r else None


def ok_izvestnyj():
    return pervyj(leak_pul()) == K.klyuch("S1", LUCH)


def test_izvestnyj_otvet_luchshij_sharp_okna_vybran():
    p = leak_pul()
    r = K.god(p, T, G[0])
    assert r["okno"] == ("2016-01-01", "2017-12-29")                  # 31.12(t−1) включительно, ни одного бара 2018
    assert r["tikery"][0]["chleny"][0] == K.klyuch("S1", LUCH)
    sh = K.sharp_okna(p.ex[:, 0, :K.granica(DATY, T)])
    assert len(r["tikery"][0]["chleny"]) == 10 and r["n_proby"] == 50
    best = sorted(range(50), key=lambda i: -sh[i])[:10]                # гладкая чаша: top-10 по Шарпу = top-10 калибратора
    assert {K.klyuch("S1", p.tochki[i]) for i in best} == set(r["tikery"][0]["chleny"])


def test_mutaciya_okno_vklyuchaet_pervoe_yanvarya_goda_t_lovitsya(monkeypatch):
    assert ok_izvestnyj()
    g = K.granica
    monkeypatch.setattr(K, "granica", lambda d, t: g(d, t) + 1)       # маска [1] в малом: первый бар года t в окне
    assert not ok_izvestnyj()


def test_mutaciya_top10_po_god_t_vmesto_t_minus_1_lovitsya(monkeypatch):
    assert ok_izvestnyj()
    monkeypatch.setattr(K, "granica", lambda d, t: year_bounds(d, t)[1])   # окно до 31.12 t (печать 1,68 / 2,99 / 1,56)
    assert not ok_izvestnyj()
    assert json.loads(pervyj(leak_pul()))["a"] in ("1.0", "2.0", "3.0")   # сдвиг к лучшей точке года t (1, 1): окно 2016–2018


OSI_P = {"a": list(range(25)), "b": list(range(5))}
SPIKE = {"a": 5, "b": 2}


def surf_plato(p):
    if p == SPIKE:
        return 3.0
    if abs(p["a"] - 5) + abs(p["b"] - 2) == 1:
        return -1.0
    return 1.5 - 0.01 * (p["a"] - 18) ** 2 - 0.05 * (p["b"] - 2) ** 2


def ok_plato_ne_pik():
    r = K.god(pul(surf_plato, lambda p: 0.5, OSI_P), T, G[0])["tikery"][0]
    return K.klyuch("S1", SPIKE) not in r["chleny"]


def test_mutaciya_plato_zamenyeno_pikom_lovitsya(monkeypatch):
    assert ok_plato_ne_pik()
    monkeypatch.setattr(SP, "plateau_value", lambda obj, nb: obj)    # № 27: плато = пик
    assert not ok_plato_ne_pik()


def test_kraj_menshe_treh_sosedej_ne_beretsya():
    ugol = {"a": 0, "b": 0}
    p = pul(lambda q: 3.0 if q == ugol else 1.0 - 0.02 * q["a"] - 0.02 * q["b"], lambda q: 0.0, OSI)
    sh, top50 = K.faza1(p, 0, 0, K.granica(DATY, T))
    assert top50[0] == p.tochki.index(ugol)
    assert p.tochki.index(ugol) not in [c["i"] for c in K.faza2(p, 0, 0, K.granica(DATY, T), sh, top50)]


def test_golosovanie_ravno_approach_a_pobajtno():
    p = pul(lambda q: 2.0 - 0.1 * ((q["a"] - 6) ** 2 + (q["b"] - 2) ** 2), lambda q: 0.0, OSI, poz=True)
    r = K.god(p, T, G[0])["tikery"][0]
    top10 = [(p.idx[kl], 0, 0) for kl in r["chleny"]]
    a = object.__new__(ApproachA)
    a.u = types.SimpleNamespace(n=DATY.size)
    a.exec_pos = lambda i, rr, e: p.poz[i, 0]
    s_t, e_t = year_bounds(DATY, T)
    assert r["poz"].tobytes() == ApproachA.ensemble(a, top10)[s_t:e_t].tobytes()


def test_sharp_okna_ravno_sr_dnevnoy():
    x = np.random.default_rng(7).standard_normal((20, 300)) * 0.01 + 0.0005
    ref = np.array([sr_dnevnoy(v) * np.sqrt(252) for v in x])
    assert np.max(np.abs(K.sharp_okna(x) - ref)) <= 1e-12
    assert K.sharp_okna(np.zeros((1, 10)))[0] == 0.0


def test_vyhod_na_konce_okna_oplachen():
    x, p = np.full((1, 5), 0.001), np.array([[0, 1, 1, 1, 1.0]])
    sh, _, ex = K.metriki(x, p, 0.0004)
    y = x.copy()
    y[0, -1] -= 0.0004
    assert sh[0] == K.sharp_okna(y)[0] and sh[0] != K.sharp_okna(x)[0] and ex[0] == 80.0


def test_daty_2020_otkaz():
    try:
        K.Pul("S1", np.array(["2019-12-31", "2020-01-03"], "datetime64[D]"), {}, [{}], np.zeros((1, 1, 2)))
    except ValueError:
        return
    raise AssertionError("даты ≥ 2020 приняты")


def test_pul_iz_ledger_i_dosschitat(tmp_path):
    rng, d = np.random.default_rng(3), DATY[:600]
    w = lm.Pisatel(tmp_path, "S1", 0, {"T": d.size, "kod": "k", "vhody": "v"})
    osi = {"a": [0.0, 1.0, 2.0, 3.0]}
    for a in osi["a"][:3]:
        kl = lm.klyuch("S1", {"a": a}, "train4", 0.0004, None)
        w.add({"status": "ok", "v": lm.V, "klyuch": kl, "klyuch_sha": lm.klyuch_sha(kl, "k", "v"), "kod": "k", "vhody": "v"},
              np.zeros(d.size), rng.standard_normal((lm.NTIK, d.size)) * 0.01)
    w.close()
    np.save(tmp_path / "S1" / "daty.npy", d)
    p = K.pul_iz_ledger(tmp_path, "S1", osi)
    assert p.ex.shape == (3, lm.NTIK, d.size) and p.n_proby() == 3 and p.tochki[1] == {"a": "1.0", "k_x": "net"}
    p.dosschitat = lambda pt: (np.zeros((lm.NTIK, d.size)), None)
    assert p.ryad(K.klyuch("S1", {"a": "3.0"}), {"a": "3.0"}, 0) is not None and p.n_proby() == 4
