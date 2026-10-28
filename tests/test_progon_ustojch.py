# -*- coding: utf-8 -*-
"""Л-10: LOYO, LOTO, бутстрап по кварталам, тикерная поправка — синтетика с известным ответом; мутации ловятся."""
import itertools
import math

import numpy as np
import pytest

from fastbt import progon_region as R, progon_ustojch as U

AX = ["a", "b", "c", "d"]
OSN = {"t_vyhod": "open2", "vhod": "0.5"}  # П-7, Ж-086: оси t_vyhod / vhod обязательны в ключе (иначе ValueError); основные значения
Q = 20


def kub(flip=(), seed=1):
    pts = [dict(zip(AX + ["n1", "n2"], v)) for v in itertools.product(range(5), range(5), range(5), range(5), (0, 1), (0, 1))]
    rows = [{"pt": {**p, **OSN}, "sdelok_y": {"2016": 40, "2017": 40}} for p in pts]
    rng = np.random.default_rng(seed)
    start = np.array(["2016-01-01"], "datetime64[M]")[0] + 3 * np.arange(16)
    mu0 = np.array([0.02 * np.exp(-sum((p[a] - 2) ** 2 for a in AX) / 2) for p in pts])
    daty, cols = [], []
    for k in range(16):
        daty.append(start[k].astype("datetime64[D]") + np.arange(Q))
        cols.append((-mu0 if k in flip else mu0)[None, :] - 0.002 + rng.normal(0, 0.005, (Q, len(pts))))
    return np.vstack(cols), rows, np.concatenate(daty)


def tik_ot(M, nt=4, nan_t0=True, seed=2):
    rng = np.random.default_rng(seed)
    tik = (M.T[:, None, :] + rng.normal(0, 0.001, (M.shape[1], nt, M.shape[0]))).astype(np.float32)
    if nan_t0:
        tik[:, 0, ::3] = np.nan
    return tik


def test_loyo_chetyre_folda_i_mediana_goda():
    M, rows, d = kub(flip=(14, 15))
    out = U.loyo(M, rows, d)
    assert [r["god"] for r in out] == [2016, 2017, 2018, 2019]
    assert all(r["sovpal"] for r in out)
    assert out[3]["mediana_kv_goda"] < out[0]["mediana_kv_goda"]  # год с перевёрнутыми кварталами оказался хуже на оставленном годе


def test_portfel_bez_nan_dni_znamenatel():
    rng = np.random.default_rng(0)
    tik = rng.normal(0, 1, (5, 4, 12)).astype(np.float32)
    tik[:, 0, ::2] = np.nan
    tik[:, 1, 1::3] = np.nan
    tik[:, 2:, 5] = np.nan  # на дату 5 у остальных нет данных, у тикера 0 тоже: после исключения 1 останется 1 тикер
    s, c = U.summy_tik(tik)
    for i in range(4):
        got = U.portfel_bez(tik, s, c, i)
        keep = [j for j in range(4) if j != i]
        with np.errstate(all="ignore"):
            exp = np.nanmean(tik[:, keep, :].astype(float), axis=1).T
        np.testing.assert_allclose(got, exp, atol=1e-12, equal_nan=True)


def test_mutaciya_loto_znamenatel_lovitsya():
    """Мутант: из числа тикеров с данными исключаемый не вычтен (знаменатель = c). Известный ответ его отличает."""
    rng = np.random.default_rng(0)
    tik = rng.normal(0, 1, (3, 4, 10)).astype(np.float32)
    tik[:, 0, ::2] = np.nan
    s, c = U.summy_tik(tik)
    exp = np.nanmean(tik[:, 1:, :].astype(float), axis=1).T
    mut = ((s - np.nan_to_num(tik[:, 0].astype(float))) / c).T
    assert not np.allclose(mut, exp)
    np.testing.assert_allclose(U.portfel_bez(tik, s, c, 0), exp, atol=1e-12)


def test_loto_17_stroka_na_tiker():
    M, rows, d = kub()
    out = U.loto(tik_ot(M), rows, d, ["T0", "T1", "T2", "T3"])
    assert [r["tiker"] for r in out] == ["T0", "T1", "T2", "T3"]
    assert all(r["sovpal"] and math.isfinite(r["celevaya"]) for r in out)


def test_butstrap_seed_vosproizvodim_i_po_kvartalam():
    M, rows, d = kub()
    ry = M[:, :50].mean(axis=1)
    a, b = U.butstrap(ry, d, n=500), U.butstrap(ry, d, n=500)
    assert a == b and a["seed"] == 20261028
    assert U.butstrap(ry, d, n=500, seed=1) != a
    sr = R._kv_sr(ry[:, None], d, list(range(16)))[0]
    v = sr[np.random.default_rng(20261028).integers(0, 16, size=(500, 16))]
    assert a["mediana_p5_50_95"] == [float(x) for x in np.percentile(np.median(v, axis=1), [5, 50, 95])]
    assert a["dolya_p5_50_95"][0] <= a["dolya_p5_50_95"][1] <= a["dolya_p5_50_95"][2]
    # мутант: те же 320 дней с возвращением (по дням) даёт другой разброс квартальных Шарпов
    rng = np.random.default_rng(20261028)
    dni = np.array([R._kv_sr(ry[rng.integers(0, len(ry), len(ry))][:, None], d, [0])[0, 0] for _ in range(50)])
    assert dni.shape == (50,) and not np.isclose(np.median(dni), a["mediana_p5_50_95"][1])


def test_butstrap_ploskiy_ryad_i_nan():
    d = np.array(["2016-01-01"], "datetime64[D]")[0] + np.arange(0, 64 * 16, 4)
    out = U.butstrap(np.zeros(len(d)), d, n=100)
    assert out["mediana_p5_50_95"] == [0.0, 0.0, 0.0] and out["dolya_p5_50_95"] == [0.0, 0.0, 0.0]


def test_vybrat_porog_nestrogo_i_gody():
    assert U.vybrat(0.2, [1, 1, 1, 1])  # «≥»: ровно 0,2 берётся
    assert not U.vybrat(0.1999, [1, 1, 1, 1])
    assert U.vybrat(0.5, [1, 1, 1, -1]) and not U.vybrat(0.5, [1, 1, -1, -1])
    assert not U.vybrat(0.5, [])


def _ex(n_days=1010, seed=0):
    d = np.array(["2016-01-01"], "datetime64[D]")[0] + np.arange(1461)
    d = d[np.is_busday(d)][:n_days]
    return d, np.random.default_rng(seed).normal(0.0005, 0.01, (3, len(d)))


def _popravka(daty, base):
    god = daty.astype("datetime64[Y]").astype(int) + 1970
    better = base.copy()
    better[0, god <= 2017] += 0.004  # тикер 0: выигрыш только в 2 годах из 4
    better[1] += 0.004  # тикер 1: во всех годах
    k = {"vhod": 1.0, "tau": 1.0}

    def ev(pt):
        return better if (pt["vhod"], pt["tau"]) == (0.5, 3.0) else base

    return U.tiker_popravka(k, ["A", "B", "C"], daty, ev, [0.5, 1.0], [1.0, 3.0]), k


def test_popravka_vyigrysh_v_dvuh_godah_ne_beretsya():
    daty, base = _ex()
    (rows, tochki), _ = _popravka(daty, base)
    r = {x["tiker"]: x for x in rows}
    assert r["A"]["vybor"] == "kandidat" and r["A"]["gody"] == 2 and r["A"]["vyigrysh"] > U.POROG
    assert r["B"]["vybor"] == "vhod=0.5;tau=3.0" and r["B"]["gody"] == 4
    assert r["C"]["vybor"] == "kandidat" and r["C"]["gody"] == 0
    assert len(tochki) == 3 and all(t["istochnik"] == "tiker" for t in tochki)


def test_mutaciya_popravka_bez_usloviya_3_iz_4(monkeypatch):
    daty, base = _ex()
    monkeypatch.setattr(U, "vybrat", lambda vy, gg: vy >= U.POROG)
    (rows, _), _ = _popravka(daty, base)
    assert rows[0]["vybor"] != "kandidat"  # мутант берёт вариант тикера A — тест выше его уронит


def test_mutaciya_porog_strogo_lovitsya(monkeypatch):
    monkeypatch.setattr(U, "vybrat", lambda vy, gg: vy > U.POROG and sum(g > 0 for g in gg) >= 3)
    assert not U.vybrat(0.2, [1, 1, 1, 1])  # мутант «>» отвергает 0,2; test_vybrat_porog_nestrogo_i_gody требует True


def test_zapisat_csv_shapka(tmp_path):
    U.zapisat_csv(tmp_path / "ustojch_loyo.csv", [dict(god=2016, sovpal=True)])
    t = (tmp_path / "ustojch_loyo.csv").read_text(encoding="utf-8").splitlines()
    assert t[0] == "# разработка 2016–2019, потолок" and t[1] == "god,sovpal" and t[2] == "2016,True"
