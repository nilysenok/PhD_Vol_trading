"""PR #505, spec МСП §11.2 п. 3 и М-Ф34: порог отбора — по рангу |score| за 252 дня внутри Sk (пул по тикерам,
окно d−252…d−1, без дня d), а не по |score|. Ранг в `reyting.rang_sily` (= `rangi_v6`, Д7).

Вход построен так, что ранг силы = 1 (sila строго растёт), поэтому score = g: значения g задают |score| напрямую.
Мутации (а) и (б) подменяют ВТОРОЙ вызов `reyting.rang_sily` (ранг |score|); первый вызов (ранг силы) не трогают.
"""
import numpy as np
import pandas as pd

from fastbt import reyting

N, DAY_A, DAY_B = 320, 300, 301
COLS = ["A", "B"]


def _vhod():
    idx = pd.bdate_range("2016-01-04", periods=N)
    rng = np.random.default_rng(5)
    g = pd.DataFrame(rng.uniform(0.5, 0.99, (N, 2)), index=idx, columns=COLS)   # история |score| < 1
    g.iloc[DAY_A, 0] = 1.0       # текущий день — максимум истории
    g.iloc[DAY_A, 1] = 0.6       # |score| = 0,6 >= 0,5, но ранг в истории низкий
    g.iloc[DAY_B, 0] = 0.99      # < максимума дня 300 (если день 300 попал в окно)
    sila = pd.DataFrame({"A": np.arange(N, dtype=float), "B": np.arange(N) + 0.5}, index=idx)
    st = pd.DataFrame(1.0, index=idx, columns=COLS)
    fr = pd.Series(np.nan, index=idx)
    fs = pd.DataFrame(np.nan, index=idx, columns=COLS)
    return st, sila, g, fr, fs


def _rank_s_tekushchim(a: pd.DataFrame, okno=252, min_nabl=20) -> pd.DataFrame:
    v = a.to_numpy(float)
    out = np.full(v.shape, np.nan)
    for d in range(len(v)):
        w = np.sort(v[max(0, d - okno):d + 1].ravel())          # ОШИБКА: день d в окне
        w = w[~np.isnan(w)]
        ok = ~np.isnan(v[d])
        if len(w) >= min_nabl:
            out[d, ok] = np.searchsorted(w, v[d, ok], side="left") / len(w)
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def _s_podmenoj(monkeypatch, vtoroy):
    """Первый вызов rang_sily — настоящий, второй (ранг |score|) — `vtoroy`."""
    nast, chet = reyting.rang_sily, [0]

    def f(x, *a, **k):
        chet[0] += 1
        return nast(x, *a, **k) if chet[0] == 1 else vtoroy(x)
    monkeypatch.setattr(reyting, "rang_sily", f)


def test_porog_po_rangu_a_ne_po_modulyu():
    st, sila, g, fr, fs = _vhod()
    sc = reyting.reyting("S1", st, sila, g, fr, fs, 0.5, porog=0.5)
    assert np.isnan(sc.iloc[DAY_A, 1])                 # |score| = 0,6 >= 0,5, но ранг ~ 0,2 < 0,5
    assert sc.iloc[DAY_A, 0] == 1.0                    # ранг 1,0 >= 0,5


def test_porog_rang_bez_tekushchego_dnya():
    st, sila, g, fr, fs = _vhod()
    sc = reyting.reyting("S1", st, sila, g, fr, fs, 0.5, porog=1.0)
    assert sc.iloc[DAY_A, 0] == 1.0                    # максимум истории: ранг ровно 1,0 только без дня d в окне
    assert np.isnan(sc.iloc[DAY_B, 0])                 # 0,99 < максимума дня 300 -> ранг < 1


def test_porog_nul_ne_menyaet_score():
    st, sila, g, fr, fs = _vhod()
    bez = reyting.reyting("S1", st, sila, g, fr, fs, 0.5)
    assert bez.iloc[DAY_A, 1] == 0.6 and bez.iloc[DAY_A, 0] == 1.0


def test_mutaciya_a_porog_po_modulyu_lovitsya(monkeypatch):
    st, sila, g, fr, fs = _vhod()
    _s_podmenoj(monkeypatch, lambda x: x)              # «ранг» = сам |score|
    sc = reyting.reyting("S1", st, sila, g, fr, fs, 0.5, porog=0.5)
    assert not np.isnan(sc.iloc[DAY_A, 1])             # мутант пропускает 0,6 -> проверка test_porog_po_rangu... упала бы


def test_mutaciya_b_rang_s_tekushchim_dnem_lovitsya(monkeypatch):
    st, sila, g, fr, fs = _vhod()
    _s_podmenoj(monkeypatch, _rank_s_tekushchim)
    sc = reyting.reyting("S1", st, sila, g, fr, fs, 0.5, porog=1.0)
    assert np.isnan(sc.iloc[DAY_A, 0])                 # мутант: ранг максимума = n/(n+1) < 1 -> проверка test_porog_rang_bez... упала бы
