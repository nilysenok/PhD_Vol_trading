"""Тестировщик PR #841: «попробуй сломать» kross_v7. Мутант = копия модуля с подменой строки исходника;
проверка обязана проходить на оригинале и падать на мутанте."""
import inspect
import types

import numpy as np
import pandas as pd
import pytest

from fastbt import kross_seksiya as ks
from fastbt import kross_v7 as kv

N, T = 8, 320
IDX = pd.bdate_range("2018-01-01", periods=T)


def _ceny(seed=1):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, N)), axis=0)), index=IDX, columns=list("abcdefgh"))
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, T))), index=IDX)
    metki = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * T, dtype=object).reshape(T, N)
    return c, m, metki


def mutant(staryj, novyj):
    src = inspect.getsource(kv)
    assert staryj in src
    mod = types.ModuleType("kv_mut")
    exec(compile(src.replace(staryj, novyj), "kv_mut", "exec"), mod.__dict__)
    return mod


# ---- нейтральность: у каждой строки обе стороны или ни одной ----
def chk_neytral(m):
    r = np.random.default_rng(7)
    sk = r.uniform(-1, 1, (200, 10))
    sk[:, 5:] = np.abs(sk[:, 5:]) * 0.9 + 0.1                       # шорта нет у половины дней
    sk[::2] = np.abs(sk[::2])                                       # чётные дни: только лонги
    ok, sg = np.ones(sk.shape, bool), np.ones(sk.shape)
    s = m.storona(sk, sg, np.full(sk.shape, "A", object), ok, cap=3)
    assert ((s == 1).any(1) == (s == -1).any(1)).all()
    assert (s[::2] == 0).all()
    w = ks.vesa(sk, sg, np.full(sk.shape, "A", object), ok)
    assert (np.sign(w).astype(np.int8)[s != 0] == s[s != 0]).all() and ((w != 0) == (s != 0)).all()
    lon, sht = np.where(w > 0, w, 0).sum(1), -np.where(w < 0, w, 0).sum(1)
    assert np.allclose(lon, sht)                                    # длинные = короткие на каждом дне


def test_neytral_ok():
    chk_neytral(kv)


def test_neytral_mutatsiya_lovitsya():
    with pytest.raises(AssertionError):
        chk_neytral(mutant("if side[1] and side[-1]:", "if side[1] or side[-1]:"))


# ---- пропуск цены: тикер выпадает, не тянется вперёд (Д4) ----
def chk_propusk(m):
    c, im, metki = _ceny()
    d, i = 280, 2
    c.iloc[d, i] = np.nan
    for sid in ("S7", "S8", "S9"):
        sk = m.panel(c, im, metki, sid)
        if sid == "S7":
            assert np.isnan(sk[d, i]) and np.isnan(sk[d + 5, i])     # день d и d+5 (окно 5) без скора
            assert np.isfinite(sk[d, [0, 1, 3]]).all()               # остальные в ранге
            ok, sg = np.ones(sk.shape, bool), np.ones(sk.shape)
            assert m.storona(sk, sg, metki, ok)[d, i] == 0
        else:
            assert np.isnan(sk[d + 21, i]) if sid == "S8" else np.isnan(sk[d, i])


def test_propusk_ok():
    chk_propusk(kv)


def test_propusk_ffill_lovitsya():
    mut = mutant_ffill_vse()
    with pytest.raises(AssertionError):
        chk_propusk(mut)


def mutant_ffill_vse():
    src = inspect.getsource(kv).replace("def panel(c: pd.DataFrame, imoex: pd.Series, metki: np.ndarray, sid: str) -> np.ndarray:",
                                        "def panel(c, imoex, metki, sid):\n    c = c.ffill()\n    return _panel(c, imoex, metki, sid)\n\n\ndef _panel(c, imoex, metki, sid):")
    mod = types.ModuleType("kv_ffill")
    exec(compile(src, "kv_ffill", "exec"), mod.__dict__)
    return mod


# ---- равные скоры: детерминированный разрыв ничьих ----
def test_nichi_determinirovany():
    sk = np.full((5, 6), 0.8)
    sk[:, 3:] = -0.8
    ok, sg = np.ones(sk.shape, bool), np.ones(sk.shape)
    metki = np.full(sk.shape, "A", object)
    a = kv.storona(sk, sg, metki, ok, cap=2)
    b = kv.storona(sk.copy(), sg.copy(), metki, ok, cap=2)
    assert np.array_equal(a, b)
    assert list(a[0]) == [1, 1, 0, -1, -1, 0]                       # при ничьей берутся меньшие индексы, cap=2
    assert np.array_equal(a != 0, ks.vesa(sk, sg, metki, ok, cap=2) != 0)


# ---- недельная ребалансировка: праздничная неделя ----
def test_nedelya_prazdnik_posledniy_torgovyy_den():
    idx = pd.DatetimeIndex(["2018-04-30", "2018-05-02", "2018-05-03",                 # пятница 4 мая — выходной в ленте
                            "2018-05-07", "2018-05-08", "2018-05-10", "2018-05-11"])
    sk = np.arange(7, dtype=float)[:, None] * np.ones((1, 3))
    out = kv.rebal(sk, idx, "nedelya")
    assert out[2, 0] == 2.0 and (out[3:6, 0] == 2.0).all() and out[6, 0] == 6.0   # решение четверга 3 мая держится всю неделю
    assert out[5, 0] == 2.0                                                        # 10 мая (чт) ещё не последний день недели


# ---- гистерезис: малое изменение ранга не меняет, большое меняет ----
def chk_gister(m):
    sk = np.array([[0.6, -0.6], [0.45, -0.45], [0.31, -0.31], [0.29, -0.29]])
    sk = np.hstack([sk, np.full((4, 2), 0.0)])
    ok, sg = np.ones(sk.shape, bool), np.ones(sk.shape)
    s = m.storona(sk, sg, np.full(sk.shape, "A", object), ok, porog=0.5, bufer=0.2)
    assert list(s[:, 0]) == [1, 1, 1, 0] and list(s[:, 1]) == [-1, -1, -1, 0]


def test_gister_ok():
    chk_gister(kv)


def test_gister_mutatsiya_lovitsya():
    with pytest.raises(AssertionError):
        chk_gister(mutant(" | ((prev == sg) & (sgs >= porog - bufer))", ""))
    with pytest.raises(AssertionError):
        chk_gister(mutant("porog - bufer", "porog + bufer"))


# ---- плацебо: сечение цело, привязка к тикеру порвана; ≠ к2 ----
def chk_placebo(f):
    r = np.random.default_rng(2)
    sk = r.uniform(-1, 1, (80, 10))
    p = f(sk, 3)
    assert np.array_equal(np.sort(sk, 1), np.sort(p, 1))             # распределение дня цело
    for j in range(10):                                              # колонка тикера не равна колонке-сдвигу исходной
        assert not any(np.array_equal(p[:, j], np.roll(sk[:, j], L)) for L in range(80))
    assert not np.array_equal(p, sk)


def test_placebo_ok():
    chk_placebo(kv.placebo_sechenie)


def test_placebo_eto_k2_lovitsya():
    with pytest.raises(AssertionError):
        chk_placebo(lambda sk, seed: kv.k2(sk, seed))


# ---- S8 и мутация «окно включает день d (и будущее)» ----
def _s8_mut(k):
    return lambda c: 2 * ks.pct(np.log(c.shift(-k) / c.shift(252))) - 1


def test_s8_ne_vidit_poslednie_21_den():
    c, im, metki = _ceny()
    d = 290
    a = kv.panel(c, im, metki, "S8")
    c2 = c.copy()
    c2.iloc[d - 20:d + 3] *= np.linspace(0.3, 3.7, N)[None, :] ** 2   # портим окно (d-20..d): пропущенный месяц
    b = kv.panel(c2, im, metki, "S8")
    assert np.array_equal(a[d], b[d], equal_nan=True)


@pytest.mark.parametrize("k", [0, 1])
def test_s8_mutatsiya_okno_s_dnem_d_lovitsya(k, monkeypatch):
    monkeypatch.setattr(ks, "skor_s8", _s8_mut(k))
    with pytest.raises(AssertionError):
        test_s8_ne_vidit_poslednie_21_den()
