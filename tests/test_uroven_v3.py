"""Синтетика для fastbt/uroven_v3.py: известные ответы и 5 мутаций (Ж-095). Мутация = неверная версия, на которой свойство обязано упасть."""
from __future__ import annotations

import numpy as np

from fastbt import uroven_v3 as uv

T = 60


def _sh_left(a):                       # значение дня d+1 в колонке d
    return np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)


# --- известные ответы ---
def test_rang_pul_maks_daet_edinicu():
    m = np.zeros((1, 40))
    m[0, 5:35] = 0.01 * np.arange(1, 31)
    r = uv.rang_pul(m)
    assert r[0, 34] == 1.0 and r[0, 3] == 0.0 and r[0, 19] == 0.0           # максимум = 1; до 20 наблюдений в пуле — 0
    assert r[0, 35] == 0.0                                                  # нет события — нуль
    m[0, 5:35] = 0.01 * np.arange(30, 0, -1)                                # убывающая: последнее — минимум пула
    r = uv.rang_pul(m)
    assert np.isclose(r[0, 34], 1 / 30) and np.isclose(r[0, 24], 1 / 20)


def test_skor_kontrtrend_storona_i_obrezka():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 0.1, (2, 300))
    c[:, 40:300:10] -= 0.9            # провалы вниз → лонг (+1)
    s = uv.skor_kontrtrend(c)
    assert (s <= 1).all() and (s >= -1).all() and s.max() == 1.0 and s.min() >= 0.0 or (s.min() > -1.0)
    ev = s[:, 100:] != 0
    assert ev.any() and (s[:, 100:][ev] > 0).sum() > (s[:, 100:][ev] < 0).sum()


def test_skor_trend_znak_proboya():
    n = 80
    c = np.concatenate([np.full(40, 100.0) + 0.01 * np.arange(40) % 3, 100 + 0.5 * np.arange(1, n - 39)])[None, :]
    s = uv.skor_trend(c, c + 0.05, c - 0.05)
    assert s.shape == c.shape and (s[:, :20] == 0).all()
    assert (s[0] >= 0).all() and (s[0] > 0).any()                           # пробой вверх: только знак +


def test_vhod_maska_porog():
    s = np.array([0.49, 0.5, -0.7, -0.3, 0.0])
    assert (uv.vhod_maska(s, 0.5) == [0, 1, -1, 0, 0]).all()


# --- мутации ---
def _novost_ok(f) -> bool:
    r, sg, v = np.zeros((1, T)), np.full((1, T), 0.01), np.full((1, T), 100.0)
    r[0, 41] = 0.1                                                          # новость в d+1 = 41
    return bool(f(r, sg, v)[0, 40] == 1.0 and f(r, sg, v)[0, 41] == 0.0)


def test_mut_kontekst_po_d_plus_1():
    assert _novost_ok(uv.g_ne_novost)
    mut = lambda r, sg, v: uv.g_ne_novost(_sh_left(r), sg, v)               # noqa: E731
    assert not _novost_ok(mut)


def _razmer_ok(f) -> bool:
    sg = np.full((1, 10), 0.02)
    a = f(sg, np.ones((1, 10)))
    sg2 = sg.copy()
    sg2[0, 5] = 0.04                                                        # меняем σ только в d+1 для d = 4
    return bool(np.isclose(f(sg2, np.ones((1, 10)))[0, 4], a[0, 4]) and f(sg2, np.ones((1, 10)))[0, 5] < a[0, 5])


def test_mut_razmer_po_sigma_d_plus_1():
    assert _razmer_ok(uv.razmer_rub)
    assert not _razmer_ok(lambda sg, s: uv.razmer_rub(_sh_left(sg), s))
    assert np.isclose(uv.razmer_rub(np.array([0.02]), np.array([-0.5]))[0], 0.003 * 200e6 / (2 * 0.02) * 0.5)


def test_mut_skor_bez_normirovki():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 0.1, (2, 300))
    c[:, 40:300:10] -= 0.9
    ok = uv.skor_kontrtrend(c)
    raw = uv.skor_kontrtrend(c, norma=False)
    ev = ok[:, 60:] != 0
    assert ev.any() and np.isclose(np.abs(ok[:, 60:][ev]).max(), 1.0)       # ранг: максимум пула = 1
    assert not np.allclose(ok, raw)                                          # сырая величина — другие числа
    assert np.abs(ok[:, 60:][ev]).mean() > 0.3 and np.abs(ok[:, 60:][ev]).min() < 1.0


def test_mut_vhod_pri_s_nizhe_poroga():
    s = np.array([0.2, 0.49])
    assert (uv.vhod_maska(s) == 0).all()
    assert (np.sign(s) != 0).all()                                           # мутация «без порога» вошла бы


def _1R_ok(f) -> bool:
    sg = np.array([0.01, 0.02, 0.03, 0.04])
    sg_b = sg.copy()
    sg_b[2] = 0.09                                                           # изменение только d+1 при d = 1
    return bool(np.isclose(f(1, 100.0, sg, 1), 100 * (1 + 2 * 0.02)) and f(1, 100.0, sg_b, 1) == f(1, 100.0, sg, 1)
                and np.isclose(f(-1, 100.0, sg, 1), 100 * (1 - 2 * 0.02)))


def test_mut_polovina_na_1R_s_dnem_d_plus_1():
    assert _1R_ok(uv.uroven_1R)
    assert not _1R_ok(lambda st, px, sg, d: uv.uroven_1R(st, px, sg, d + 1))


def test_likvidnost_5_procentov():
    ob = np.full((1, 30), 1e9)
    assert uv.g_likvidnost(np.full((1, 30), 5e7), ob)[0, 25] == 1.0
    assert uv.g_likvidnost(np.full((1, 30), 5.1e7), ob)[0, 25] == 0.0
    assert uv.g_likvidnost(np.full((1, 30), 1.0), ob)[0, 5] == 0.0         # меньше 20 дней — nan, не ликвидно
