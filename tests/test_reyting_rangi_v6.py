"""Д7: ранг силы в рейтинге МСП — одна функция `rangi_v6` (пул по тикерам), а не вторая копия.

`_staraya` — прежняя локальная реализация `reyting.rang_sily` (до V6-c), оставлена здесь только как эталон сверки.
"""
import numpy as np
import pandas as pd

from fastbt import rangi_v6, reyting


def _staraya(sila, okno=252, min_nabl=20):
    a = sila.to_numpy(float)
    out = np.full(a.shape, np.nan)
    for i in range(len(a)):
        w = a[max(0, i - okno):i].ravel()
        w = np.sort(w[~np.isnan(w)])
        ok = ~np.isnan(a[i])
        if len(w) >= min_nabl:
            out[i, ok] = np.searchsorted(w, a[i, ok], side="left") / len(w)
    return pd.DataFrame(out, index=sila.index, columns=sila.columns)


def _panel(n=500, k=4, seed=3, nan_dolya=0.3):
    rng = np.random.default_rng(seed)
    a = np.round(rng.random((n, k)), 2) + 0.1             # округление — повторы (проверка «строго меньше»)
    a[rng.random((n, k)) < nan_dolya] = np.nan
    return pd.DataFrame(a, index=pd.bdate_range("2016-01-04", periods=n), columns=list("ABCD")[:k])


def test_sovpadaet_so_staroj_realizaciej():
    for seed, dolya, okno, mn in [(3, 0.3, 252, 20), (4, 0.0, 252, 20), (5, 0.8, 252, 20), (6, 0.3, 50, 5)]:
        s = _panel(seed=seed, nan_dolya=dolya)
        novaya, staraya = reyting.rang_sily(s, okno, mn), _staraya(s, okno, mn)
        assert novaya.index.equals(s.index) and list(novaya.columns) == list(s.columns)
        assert (novaya.isna().to_numpy() == staraya.isna().to_numpy()).all()
        assert np.nanmax(np.abs(novaya.to_numpy() - staraya.to_numpy())) <= 1e-12
        assert novaya.notna().to_numpy().sum() > 0


def test_ravno_rangi_v6_obshchiy_s_transponirovaniem():
    s = _panel()
    ozhid = rangi_v6.rang_obshchiy(s.to_numpy(float).T).T
    np.testing.assert_array_equal(reyting.rang_sily(s).to_numpy(), ozhid)


def test_mutaciya_pul_po_tikeru_lovitsya(monkeypatch):
    """Подмена пула «все тикеры» на «по тикеру» меняет результат — сверка со старой реализацией это ловит."""
    s = _panel()
    monkeypatch.setattr(rangi_v6, "rang_obshchiy", rangi_v6.rang_tikera)
    r, e = reyting.rang_sily(s), _staraya(s)
    assert not (r.isna().to_numpy() == e.isna().to_numpy()).all() or np.nanmax(np.abs(r.to_numpy() - e.to_numpy())) > 1e-3
