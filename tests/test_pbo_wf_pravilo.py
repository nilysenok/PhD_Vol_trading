"""К-wf6, лист 2: правило процедуры внутри CSCV (Ж-080), мутация № 42 «argmax вместо правила процедуры»."""
import numpy as np

from fastbt import pbo_wf as W

ND, P = 63, 12
CHAIN = [[j for j in (i - 1, i + 1) if 0 <= j < P] for i in range(P)]     # ось из 12 точек, соседи ±1


def sums(x):
    return x.sum(axis=2), (x ** 2).sum(axis=2), np.full(x.shape[1], float(x.shape[2]))


def sintetika(seed=3):
    """Точки 4, 5, 6 — плато (стабильный плюс); точка 1 — пик: сильно плюс только в блоке 0, минус дальше; остальные —
    слегка минус. Известный ответ: argmax берёт пик, когда блок 0 in-sample (126 из 252 разбиений), и на out-of-sample
    пик хуже всех -> PBO argmax ≈ 0,5; правило процедуры берёт плато 4–6 (плато пика — среднее с отрицательными соседями)."""
    x = np.random.default_rng(seed).normal(0.0, 0.01, (P, W.NQ, ND)) - 0.0005
    x[4:7] += 0.0020
    x[1, :5] += 0.0300                                 # блок 0 = кварталы 0–4
    x[1, 5:] -= 0.0030
    return sums(x)


def pravilo():
    return W.pravilo_procedury(CHAIN, top_f1=6, top_f2=3, ens_k=3, min_nb=2)


def test_sintetika_pravilo_i_argmax_izvestny():
    r = W.pbo_wf(*sintetika(), pravilo())
    a = W.pbo_wf(*sintetika(), W.vybor_argmax)
    assert r["pbo"] <= 0.05 and r["n_propusk"] == 0
    assert a["pbo"] >= 0.30


def test_mutatsiya_42_podmena_pravila_na_argmax_lovitsya():
    """№ 42: правило внутри CSCV заменено на argmax -> PBO обязан сдвинуться > 0,25."""
    est = W.pbo_wf(*sintetika(), pravilo())["pbo"]
    mut = W.pbo_wf(*sintetika(), lambda sh, tr: W.vybor_argmax(sh, tr))["pbo"]
    assert mut - est > 0.25


def test_pravilo_vybiraet_plato_a_ne_pik():
    S, Q2, nb = sintetika()
    Sb, Qb, nbb = W.bloki(S, Q2, nb)
    tr = [0, 1, 2, 3, 4]                               # сильные блоки пика in-sample
    sh = W.sharpe_iz_summ(Sb[:, tr].sum(1), Qb[:, tr].sum(1), nbb[tr].sum())
    assert int(np.argmax(sh)) == 1
    assert set(pravilo()(sh, tuple(tr))) <= {3, 4, 5, 6, 7} and 1 not in pravilo()(sh, tuple(tr))


def test_shum_pravilo_pbo_okolo_polovinu():
    """252 разбиения одной выборки сильно зависимы: среднее по 12 выборкам, а не одна."""
    sos = [[j for j in (i - 1, i + 1) if 0 <= j < 300] for i in range(300)]
    pbos = [W.pbo_wf(*sums(np.random.default_rng(s).normal(0.0, 0.01, (300, W.NQ, ND))),
                     W.pravilo_procedury(sos, min_nb=2))["pbo"] for s in range(12)]
    assert abs(float(np.mean(pbos)) - 0.5) <= 0.12


def test_gran_kray_bez_sosedey_propusk_i_nikto_ne_proshyol():
    sh = np.array([0.5, 0.4, 0.3])
    assert W.pravilo_procedury([[], [], []])(sh, ()) is None                  # все — край
    neg = np.array([-1.0, -2.0, -3.0, -0.5])
    sos = [[1, 2, 3]] * 4
    assert W.pravilo_procedury(sos, ens_k=2, min_nb=3)(neg, ()) == [3, 0]     # ворота никто — top ens_k по Шарпу


def test_gran_nichya_menshiy_indeks():
    sh = np.ones(4)
    assert W.pravilo_procedury([[1, 2, 3]] * 4, ens_k=2, min_nb=3)(sh, ()) == [0, 1]
