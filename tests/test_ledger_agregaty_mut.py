"""Тестировщик PR #910: «попробуй сломать» агрегаты v7 и мутации, не пойманные tests/test_ledger_agregaty.py (№ 70: импорты без tests.)."""
import math

import numpy as np
import pandas as pd
import pytest

import fastbt.agregaty_v7 as A
from fastbt.metriki_e import msr, nedelnye_summy, sharpe

DATY = np.arange(np.datetime64("2017-01-02"), np.datetime64("2026-03-01"))
DATY = DATY[np.is_busday(DATY)]
T = DATY.size
RNG = np.random.default_rng(11)
X = RNG.normal(3e-4, 1e-2, T)
Z = np.zeros(T)
REZ = RNG.integers(0, 3, T)
GOD = DATY.astype("datetime64[Y]").astype(int) + 1970


def test_itog_2017_2025_iz_summ_raven_ryadu():
    ag = A.agregaty({"agr": A.sobrat(DATY, X, Z, Z, REZ)})
    n, s, q = (sum(A.sobrat(DATY, X, Z, Z, REZ)["d"][i][:9]) for i in range(3))
    assert A.sharpe_iz_summ(n, s, q) == pytest.approx(sharpe(X[GOD <= 2025]), abs=1e-12)
    assert ag["sharpe_itog"] == pytest.approx(sharpe(X), abs=1e-12)


def test_nulevaya_dispersiya_bez_inf_nan_v_sharpe():
    x = np.full(T, 1e-3)
    ag = A.agregaty({"agr": A.sobrat(DATY, x, Z, Z, REZ)})
    assert ag["sharpe_itog"] == sharpe(x) == 0.0                       # как metriki_e: 0, не inf и не nan
    assert all(math.isfinite(v) for v in ag["sharpe_gody"].values())


def test_msr_vne_oblasti_kornisha_fishera_daet_nan_a_ne_inf():
    w = np.full(5, 0.01)                                                  # константа: sd = 0, mVaR < 0
    r = A.msr_iz_summ(5, w.sum(), (w**2).sum(), (w**3).sum(), (w**4).sum())
    assert r != r                                                         # nan
    # ЗАМЕЧАНИЕ: пометки «вне области КФ» (z_cf ≥ 0 → mVaR ≤ 0) в агрегатах нет; различить причину nan нельзя


def test_msr_iz_momentov_raven_msr_nedeli_na_malyh_vyborkah():
    for n in (3, 4, 10):
        w = RNG.normal(0.002, 0.02, n)
        got = A.msr_iz_summ(n, *(float((w**p).sum()) for p in (1, 2, 3, 4)))
        ref = msr(w, 52)
        assert (got != got and ref != ref) or got == pytest.approx(ref, rel=1e-8)


def test_mutaciya_ddof0_lovitsya(monkeypatch):
    monkeypatch.setattr(A, "sqrt", lambda v: math.sqrt(v * (T - 1) / T))   # грубо ломает дисперсию
    ag = A.agregaty({"agr": A.sobrat(DATY, X, Z, Z, REZ)})
    with pytest.raises(AssertionError):
        assert ag["sharpe_gody"][2018] == pytest.approx(sharpe(X[GOD == 2018]), abs=1e-12)


def test_mutaciya_god_nedeli_po_pervomu_dnyu_lovitsya(monkeypatch):
    """Год недели по первому дню вместо последнего: недели через границу года уходят в другой год."""
    iz = pd.Series(X, index=pd.DatetimeIndex(DATY))
    gr = iz.groupby(iz.index.to_period("W-FRI"))
    w = gr.sum().to_numpy()
    pervyj = gr.apply(lambda s: s.index[0].year).to_numpy()
    posl = gr.apply(lambda s: s.index[-1].year).to_numpy()
    assert (pervyj != posl).any()                                         # граница года на неделе существует
    ag = A.agregaty({"agr": A.sobrat(DATY, X, Z, Z, REZ)})
    for g in (2018, 2019, 2021):
        assert ag["msr_ned_gody"][g] == pytest.approx(msr(w[posl == g], 52), rel=1e-8)
    g = int(pervyj[pervyj != posl][0])                                    # год, где наборы недель различаются
    assert w[pervyj == g].size != w[posl == g].size or not np.allclose(w[pervyj == g], w[posl == g])
    assert ag["msr_ned_gody"][g] != pytest.approx(msr(w[pervyj == g], 52), rel=1e-8)


def test_nedeli_summa_po_godam_sovpadaet_s_nedelnymi_summami():
    a = A.sobrat(DATY, X, Z, Z, REZ)["w"]
    assert sum(a[0]) == len(nedelnye_summy(X, DATY))
    assert sum(a[1]) == pytest.approx(X.sum(), abs=1e-9)
