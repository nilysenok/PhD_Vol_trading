"""Семья E, подвыборка топ-10 по обороту: ex-ante (год Y−1), состав фиксирован на год (§8 п. 11). Синтетика."""
import numpy as np
import pandas as pd
import pytest

from fastbt import likvidnost_e as L


def _ob():
    rng = np.random.default_rng(3)
    rows = []
    for y in (2017, 2018, 2019, 2020):
        v = rng.permutation(np.arange(1, 18)) * 1e9
        rows += [{"ticker": t, "year": y, "oborot_rub": float(x)} for t, x in zip(L.TICKERS, v)]
    return pd.DataFrame(rows)


def _top_tekushchiy_god(year, ob):
    """Мутация §8 п. 11: оборот ТЕКУЩЕГО года."""
    p = ob[ob.year == year].sort_values("oborot_rub", ascending=False)
    return sorted(p.ticker.head(10))


def _top_skolzyashchiy(date, ob):
    """Мутация: состав пересчитывается внутри года (по предыдущему году, но сдвигается с месяцем)."""
    d = pd.Timestamp(date)
    y = d.year - 1 + (1 if d.month > 6 else 0)
    return _top_tekushchiy_god(y, ob)


def test_top10_po_oborotu_proshlogo_goda():
    ob = _ob()
    for y in (2018, 2019, 2020):
        p = ob[ob.year == y - 1].sort_values("oborot_rub", ascending=False)
        assert L.top10(y, ob) == sorted(p.ticker.head(10)) and len(L.top10(y, ob)) == 10


def test_mutaciya_top10_tekushchego_goda_lovitsya():
    ob = _ob()
    # на случайной синтетике составы Y−1 и Y различаются; мутация обязана дать другой результат хотя бы в одном из лет
    assert any(_top_tekushchiy_god(y, ob) != L.top10(y, ob) for y in (2018, 2019, 2020))


def test_tekushchiy_god_ne_vliyaet():
    ob = _ob()
    base = L.top10(2019, ob)
    mod = ob.copy()
    mod.loc[(mod.year == 2019) & (mod.ticker == base[0]), "oborot_rub"] = 0.0   # оборот Y не должен менять состав Y
    assert L.top10(2019, mod) == base


def test_sostav_ne_menyaetsya_vnutri_goda():
    ob = _ob()
    dates = pd.bdate_range("2019-01-01", "2019-12-31")
    assert len({tuple(L.sostav_na_datu(d, ob)) for d in dates}) == 1


def test_mutaciya_sostav_vnutri_goda_lovitsya():
    ob = _ob()
    dates = pd.bdate_range("2019-01-01", "2019-12-31")
    assert len({tuple(_top_skolzyashchiy(d, ob)) for d in dates}) > 1   # такой детектор (число составов = 1) её ловит


def test_net_dyry_2017_net_tikera():
    ob = _ob()
    with pytest.raises(AssertionError):
        L.top10(2018, ob[~((ob.year == 2017) & (ob.ticker == "SBER"))])
    with pytest.raises(AssertionError):
        L.oborot_po_godam(years=(2022,), cache=False)


def test_2017_est_po_vsem_17_realnye_dannye():
    if not (L.BARS1M / "SBER_2017.csv").exists():
        pytest.skip("нет bars1m")
    t = L.oborot_po_godam()
    assert set(t[t.year == 2017].ticker) == set(L.TICKERS) and (t.oborot_rub > 0).all()
