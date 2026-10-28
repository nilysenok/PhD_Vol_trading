# -*- coding: utf-8 -*-
"""Ж-080: PBO по CSCV с выбором по правилу этапа. Мутация «правило внутри CSCV подменено на argmax» обязана менять PBO."""
import functools
import math

import numpy as np
import pytest

from fastbt import etap1_poisk as P
from fastbt import etap1_poisk_vorota as V
from fastbt import metriki_e as ME
from fastbt import pbo_pravilo as B
from fastbt import pbo_pravilo_sprav as S
from fastbt import sweep_drivers as sd

T, NB = 240, 30
DATES = np.concatenate([np.datetime64("2018-01-01") + np.arange(120), np.datetime64("2019-01-01") + np.arange(120)])


def zig(m, h, n=T):
    return m + h * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def sintetika():
    """x = 0…9 на одной оси. 4, 5, 6 — плато (стабильно +0,1 %), x = 1 — пик: в блоке 0 даёт +5 %/день, в остальных −1 %,
    остальные точки −0,5 %. Известный ответ: argmax берёт пик, когда блок 0 in-sample (35 из 70 разбиений), и там пик на
    out-of-sample хуже всех -> PBO argmax = 0,5; центр плато — всегда x = 5, он выше 6 из 9 точек -> PBO = 0."""
    rows = [zig(-0.005, 0.01) for _ in range(10)]
    for x in (4, 5, 6):
        rows[x] = zig(0.001, 0.01)      # разброс обязателен: при нулевом хвосте mVaR ≤ 0 и mSR = nan (метрика не определена)
    spike = zig(-0.01, 0.02)
    spike[:NB] = zig(0.05, 0.03, NB)
    rows[1] = spike
    M = np.array(rows)
    coords = [{"x": x} for x in range(10)]
    kub = B.postroit_kub(coords, {"x": list(range(10))}, ["x"], {})
    return M, coords, functools.partial(B.vybor_centr_plato, kub=kub, dop=np.ones(10, bool))


def test_sintetika_argmax_i_centr_plato_izvestny():
    M, coords, centr = sintetika()
    assert B.pbo_cscv_pravilo(M, coords, B.vybor_argmax, dates=DATES) == 0.5
    assert B.pbo_cscv_pravilo(M, coords, centr, dates=DATES) == 0.0
    assert B.cscv(M, coords, centr, dates=DATES)["n_propusk"] == 0


def test_mutatsiya_podmena_pravila_na_argmax_menyaet_pbo():
    M, coords, centr = sintetika()
    est = B.pbo_cscv_pravilo(M, coords, centr, dates=DATES)
    mut = B.pbo_cscv_pravilo(M, coords, lambda Mi, c, d: B.vybor_argmax(Mi, c, d), dates=DATES)
    assert abs(est - mut) > 0.4


def etalon_3x2():
    """Эталон проверяющего (docs/otchety/2026-10-28_proverka_pbo_pravilo.md): сетка 3×2, T = 8, блок = строка, 70 разбиений;
    c0…c4 — константы 0,8 / 0,5 / 0,6 / 0,7 / 0,9, c5 — шип +10 (строки 0–3) и −10 (строки 4–7). Целевая — среднее."""
    pts = [{"a": a, "b": b} for a in range(3) for b in range(2)]
    M = np.array([[v] * 8 for v in (0.8, 0.5, 0.6, 0.7, 0.9)] + [[10.0] * 4 + [-10.0] * 4])
    kub = B.postroit_kub(pts, {"a": [0, 1, 2], "b": [0, 1]}, ["a", "b"], {})
    mean = lambda X, d, rows: X[rows].mean(axis=1)                                            # noqa: E731
    return M, pts, functools.partial(B.vybor_centr_plato, kub=kub, dop=np.ones(6, bool), obj_fn=mean)


def test_etalon_proveryayushchego_3x2_17_iz_70_i_0_iz_70():
    M, pts, centr = etalon_3x2()
    a, c = B.cscv(M, pts, B.vybor_argmax), B.cscv(M, pts, centr)
    assert (round(a["pbo"] * 70), a["n"]) == (17, 70) and (round(c["pbo"] * 70), c["n"], c["n_propusk"]) == (0, 70, 0)
    assert a["pbo"] == 17 / 70 and c["pbo"] == 0.0


def test_etalon_3x2_mutatsiya_argmax_vnutri_centra_daet_17_vmesto_0():
    M, pts, _ = etalon_3x2()
    assert B.cscv(M, pts, lambda Mi, c, d: B.vybor_argmax(Mi, c, d))["pbo"] == 17 / 70 != 0.0


def test_g1_beryot_centr_plato_a_argmax_spravochnyy(tmp_path, monkeypatch):
    """Итог поиска: обе колонки; G1_pbo — по `pbo` (центр плато). Мутация «G1 по argmax» (pbo без pravilo) совпадает с pbo_argmax."""
    from test_etap1_poisk import SETKA, World
    fake = lambda M, nblk=V.NBLK, pravilo=None: 0.9 if pravilo is None else 0.05              # noqa: E731  правило -> 0,05, argmax -> 0,9
    monkeypatch.setattr(P, "pbo", fake)
    row = P.run_strategy("S9", World().evaluate, lambda p: True, out=tmp_path / "a", run_id="1", setka=SETKA)
    assert (row["pbo"], row["pbo_argmax"], row["G1_pbo"]) == (0.05, 0.9, True)
    monkeypatch.setattr(P, "pbo", lambda M, nblk=V.NBLK, pravilo=None: fake(M, nblk))          # мутация: G1 по argmax
    mut = P.run_strategy("S9", World().evaluate, lambda p: True, out=tmp_path / "b", run_id="1", setka=SETKA)
    assert (mut["pbo"], mut["G1_pbo"]) == (0.9, False) and mut["pbo"] != row["pbo"]


def test_vorota_pbo_s_pravilom_i_bez():
    """V.pbo(M) без pravilo — прежний argmax (справочная колонка), с pravilo — правило этапа; эти два числа различны."""
    M, coords, centr = sintetika()
    prav = dict(coords=coords, dates=DATES, select_fn=centr)
    assert V.pbo(M.T) == 0.5 and V.pbo(M.T, pravilo=prav) == 0.0
    assert math.isnan(V.pbo(M.T[:, :1], pravilo=prav))


def test_pravilo_etapa_stroit_stolbcy_kak_matrix():
    """Ledger из двух точек оси x + ансамбль: столбцы pravilo_etapa в порядке V.matrix, у ансамбля coords None."""
    rng = np.random.default_rng(5)
    res = lambda: {"excess_d": {2018: rng.normal(0, 0.01, 130), 2019: rng.normal(0, 0.01, 110)}, "sdelok_y": {2018: 40, 2019: 40},  # noqa: E731
                   "storony_y": {2018: 80, 2019: 80}}
    led = V.Ledger(None)
    for x in (0, 1):
        led.cache[(V.pkey({"x": x}), "train")] = ({"x": x}, res())
    led.cache[(("ansambl", "a", "b"), "train")] = ({}, res())
    pr = B.pravilo_etapa(led, {"x": [0, 1]}, ["x"], {}, {"x": 0})
    assert pr["coords"] == [{"x": 0}, {"x": 1}, None] and len(pr["dates"]) == V.matrix(led).shape[0] == 240
    assert pr["select_fn"].keywords["dop"].tolist() == [True, True, False]


def test_argmax_ravno_staromu_pbo_cscv():
    M = np.random.default_rng(7).normal(0, 0.01, (40, 240))
    assert B.pbo_cscv_pravilo(M, [None] * 40, B.vybor_argmax) == sd.pbo_cscv(M.T.tolist(), 8)["pbo"]


@pytest.mark.parametrize("sid,it", [("S5", 1), ("S6", 1), ("S5", 4)])
def test_argmax_na_realnoy_matritse_ravno_prezhnemu(sid, it):
    led = S.nayti_ledger()
    if led is None:
        pytest.skip("ledger (вне git) не найден: S.nayti_ledger() ищет docs/tables/zh051/_ledger в копии и предках")
    r = S.schitat(sid, it, led)
    assert r["argmax_sovpal"] and r["razbienij"] == 70 and r["propusk_centr"] == 0


def test_pustaya_i_korotkaya_lenta_nan():
    assert math.isnan(B.pbo_cscv_pravilo(np.zeros((5, 4)), [None] * 5, B.vybor_argmax))
    assert math.isnan(B.pbo_cscv_pravilo(np.zeros((1, 240)), [None], B.vybor_argmax))


def test_nulevoy_obyom_net_kandidata_razbienie_propuskaetsya():
    """Все ряды нули: mSR = nan везде -> кандидатов нет -> каждое разбиение пропущено, PBO = nan, а не ложный 0."""
    kub = B.postroit_kub([{"x": x} for x in range(5)], {"x": list(range(5))}, ["x"], {})
    f = functools.partial(B.vybor_centr_plato, kub=kub, dop=np.ones(5, bool))
    r = B.cscv(np.zeros((5, 240)), [{"x": x} for x in range(5)], f, dates=DATES)
    assert r["n"] == 0 and r["n_propusk"] == 70 and math.isnan(r["pbo"])


def test_perepytannyy_znak_menyaet_pbo_argmax():
    M, coords, _ = sintetika()
    assert B.pbo_cscv_pravilo(-M, coords, B.vybor_argmax, dates=DATES) != 0.5


def test_msr_stolbcy_ravno_metriki_e():
    X = np.random.default_rng(3).standard_t(4, (200, 6)) * 0.01 + np.linspace(0, 0.002, 6)
    assert np.allclose(B.msr_stolbcy(X), [ME.msr(X[:, j]) for j in range(6)], rtol=1e-12, equal_nan=True)


def test_god_s_menee_21_dnya_v_polovine_ne_vhodit_v_min():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, (1, 60))
    d = np.concatenate([np.datetime64("2018-01-01") + np.arange(45), np.datetime64("2019-01-01") + np.arange(15)])
    assert np.isclose(B.min_msr(x, d)[0], ME.msr(x[0, :45]))
    d2 = np.concatenate([np.datetime64("2018-01-01") + np.arange(30), np.datetime64("2019-01-01") + np.arange(30)])
    assert np.isclose(B.min_msr(x, d2)[0], min(ME.msr(x[0, :30]), ME.msr(x[0, 30:])))


def test_konstanty_ravny_etapu():
    assert (B.TIE, B.WIN, B.TOPK, B.Z95) == (V.TIE, V.WIN, P.TOPK, ME.Z95)


def test_centr_na_polnom_periode_ravno_pick_by_plateau():
    """Двумерная сетка 5×4, случайные ряды, «половина» = весь период: тот же центр, что sd.pick_by_plateau по целевой min mSR."""
    rng = np.random.default_rng(11)
    ax = {"a": list(range(5)), "b": list(range(4))}
    pts = [{"a": a, "b": b} for a in ax["a"] for b in ax["b"]]
    M = rng.normal(0.0008, 0.01, (20, 240)) + rng.normal(0, 0.0006, (20, 1))
    obj = B.min_msr(M, DATES)
    fact = dict(rows=[{**p, "objective": float(o)} for p, o in zip(pts, obj)], keys=["a", "b"], grid=ax)
    top = next(r for r in sd.pick_by_plateau(fact, k=20, radius=1, min_neighbors=3) if not r["edge"] and math.isfinite(r["objective"]))
    kub = B.postroit_kub(pts, ax, ["a", "b"], {})
    j = B.vybor_centr_plato(M, pts, DATES, kub=kub, dop=np.ones(20, bool), tie=0.0)
    assert pts[j] == {k: top[k] for k in ("a", "b")}
