"""Тестировщик #734: случаи, которых нет в test_wf_vneshnie.py (реальные данные: 12.06.2024 — выходной, CNY без бара)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sigma_real, vneshnie_usdrub as vu

D = pd.Timestamp


def _bary(dni, vol, start=90.0):
    ts = [D(d) + pd.Timedelta(hours=10, minutes=10 * k) for d in dni for k in range(12)]
    c = start * np.exp(np.cumsum(np.tile([vol, -vol], len(ts) // 2)))
    return pd.DataFrame({"timestamp": ts, "close": c, "open": c, "high": c, "low": c, "volume": 1})


def test_styk_s_vyhodnym_12_06_pervaya_dohodnost_cny_13_06():
    ix = pd.DatetimeIndex(["2024-06-10", "2024-06-11", "2024-06-13", "2024-06-14"])
    usd = pd.Series([88.6, 89.2], index=ix[:2])
    cny = pd.Series([12.17, 12.24, 12.20, 12.16], index=ix)
    dx = vu.dx_usdrub(usd, cny)
    assert D("2024-06-12") not in dx.index
    assert dx[D("2024-06-13")] == pytest.approx(np.log(12.20 / 12.24))      # из CNY, 11.06 -> 13.06, не через USD
    assert dx[D("2024-06-11")] == pytest.approx(np.log(89.2 / 88.6))         # из USD, не из CNY
    assert np.isfinite(dx.loc["2024-06-11":]).all() and not dx.index.duplicated().any()


def test_cny_nachinaetsya_12_06_pervaya_dohodnost_nan_bez_podstanovki():
    usd = pd.Series([88.6, 89.2], index=pd.DatetimeIndex(["2024-06-10", "2024-06-11"]))
    cny = pd.Series([12.20, 12.16], index=pd.DatetimeIndex(["2024-06-12", "2024-06-13"]))
    dx = vu.dx_usdrub(usd, cny)
    assert np.isnan(dx[D("2024-06-12")])                                      # не ln(CNY/USD) и не 0
    assert dx[D("2024-06-13")] == pytest.approx(np.log(12.16 / 12.20))


def test_rv_odin_den_na_styke_bez_dublya_i_sigma_okno_smeshivaet_rv():
    dni = list(pd.bdate_range("2024-04-01", "2024-06-21"))
    usd = _bary([d for d in dni if d <= D("2024-06-11")], 0.001)
    cny = _bary(dni, 0.01)
    rv = vu.rv_usdrub(usd, cny)
    assert not rv.index.duplicated().any()
    assert rv[D("2024-06-11")] == pytest.approx(sigma_real.rv_1850(usd)[D("2024-06-11")])   # день стыка — USD
    assert rv[D("2024-06-12")] == pytest.approx(sigma_real.rv_1850(cny)[D("2024-06-12")])   # после — CNY
    _, sg = vu.faktor_usdrub(usd, cny)
    # σ22 в окне после стыка = среднее rv обоих рядов (rv — уже дневная величина из доходностей своего ряда)
    assert sg[D("2024-06-12")] == pytest.approx(np.sqrt(rv.loc[:"2024-06-12"].tail(22).mean()))
    assert sg[D("2024-06-12")] > np.sqrt(rv.loc[:"2024-06-11"].tail(22).mean())
