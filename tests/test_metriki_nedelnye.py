"""Недельные метрики (Ж-124 п. 3): известный ответ на синтетике + мутации, которые тест обязан поймать (mutations.md, Д7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import metriki_e as me

W = np.array([0.012, -0.020, 0.031, 0.004, -0.007, 0.018, -0.011, 0.025, 0.002, -0.015, 0.009, 0.020])
Z = -1.645


def dni(w=W, start="2024-01-01"):
    """Торговые дни Пн–Пт; в неделе i сумма дневных = w[i] (пять дней: w/5 + возмущения, сумма нуля)."""
    pert = np.array([0.003, -0.001, 0.002, -0.004, 0.0])
    days = pd.bdate_range(start, periods=5 * len(w))
    x = np.concatenate([wi / 5 + pert for wi in w])
    return x, days.to_numpy()


def ozhidaemoe(w=W) -> dict:
    """Независимый ответ: формулы выписаны здесь, в код проекта не заходят."""
    m, sd, d = w.mean(), w.std(ddof=1), w - w.mean()
    s, k = (d**3).mean() / (d**2).mean() ** 1.5, (d**4).mean() / (d**2).mean() ** 2 - 3
    zcf = Z + (Z**2 - 1) * s / 6 + (Z**3 - 3 * Z) * k / 24 - (2 * Z**3 - 5 * Z) * s**2 / 36
    neg = w[w < 0]
    return dict(sharpe_ned=m / sd * np.sqrt(52), sortino_ned=m * 52 / (neg.std(ddof=1) * np.sqrt(52)),
                msr_ned=m * 52 / (-(m + zcf * sd) * np.sqrt(52)), skew_ned=s, kurt_ned=k, n_nedel=len(w))


def proveri(res: dict) -> None:
    for k, v in ozhidaemoe().items():
        assert res[k] == pytest.approx(v, rel=1e-9), k


def test_znanyj_otvet():
    x, d = dni()
    assert np.allclose(me.nedelnye_summy(x, d), W, atol=1e-12)
    proveri(me.metriki_ned(x, d))


def test_nepolnye_nedeli_i_prazdnaya_nedelya():
    # среда → пятница (3 дня), пропущенная неделя целиком (праздник), потом Пн, Сб: суббота относится к неделе, закрывающейся следующей пятницей
    d = np.array(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-15", "2024-01-20", "2024-01-22"], "datetime64[D]")
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    assert me.nedelnye_summy(x, d).tolist() == [7.0, 8.0, 48.0]


def test_dolnye_dni_v_nedele_poryadok_vhoda_ne_vazhen():
    x, d = dni()
    p = np.random.default_rng(1).permutation(len(x))
    assert np.allclose(me.nedelnye_summy(x[p], d[p]), W, atol=1e-12)


def test_vne_oblasti_nan_a_ne_chislo():
    w = np.array([0.0101, 0.0099] * 6)               # mVaR < 0 (среднее выше z·sd): формула не определена
    x, d = dni(w)
    r = me.metriki_ned(x, d)
    assert np.isnan(r["msr_ned"]) and np.isfinite(r["sharpe_ned"])
    assert np.isnan(me.msr(w, 52))


def test_mutaciya_nedelya_po_W_MON_lovitsya(monkeypatch):
    def mut(r, daty):
        s = pd.Series(np.asarray(r, float), index=pd.DatetimeIndex(daty))
        return s.groupby(s.index.to_period("W-MON")).sum().to_numpy()
    x, d = dni()
    monkeypatch.setattr(me, "nedelnye_summy", mut)
    sums = me.nedelnye_summy(x, d)
    assert not (len(sums) == len(W) and np.allclose(sums, W, atol=1e-12))     # 13 недель вместо 12
    with pytest.raises(AssertionError):
        proveri(me.metriki_ned(x, d))


def test_mutaciya_sdvig_dnya_sosednej_nedeli_lovitsya():
    x, d = dni()
    d2 = d.copy()
    d2[5] = d[4]                                      # первый день недели 2 переехал на пятницу недели 1
    with pytest.raises(AssertionError):
        proveri(me.metriki_ned(x, d2))


def test_mutaciya_sqrt252_vmesto_sqrt52_lovitsya(monkeypatch):
    x, d = dni()
    monkeypatch.setattr(me, "NED", 252)
    with pytest.raises(AssertionError):
        proveri(me.metriki_ned(x, d))


def test_mutaciya_ddof0_lovitsya(monkeypatch):
    x, d = dni()
    monkeypatch.setattr(me, "sr_dnevnoy", lambda r: float(np.mean(r) / np.std(r, ddof=0)))
    with pytest.raises(AssertionError):
        proveri(me.metriki_ned(x, d))


def test_mutaciya_msr_ned_podmenen_dnevnym_lovitsya(monkeypatch):
    x, d = dni()
    orig = me.msr
    monkeypatch.setattr(me, "msr", lambda r, g=252: orig(r))           # шкала g проигнорирована → 252
    with pytest.raises(AssertionError):
        proveri(me.metriki_ned(x, d))


def test_dnevnye_msr_i_sortino_ne_izmenilis():
    x, _ = dni()
    m, sd, d = x.mean(), x.std(ddof=1), x - x.mean()
    s, k = (d**3).mean() / (d**2).mean() ** 1.5, (d**4).mean() / (d**2).mean() ** 2 - 3
    zcf = Z + (Z**2 - 1) * s / 6 + (Z**3 - 3 * Z) * k / 24 - (2 * Z**3 - 5 * Z) * s**2 / 36
    assert me.msr(x) == pytest.approx(m * 252 / (-(m + zcf * sd) * np.sqrt(252)), rel=1e-12)
    neg = x[x < 0]
    assert me.sortino_e(x) == pytest.approx(m * 252 / (neg.std(ddof=1) * np.sqrt(252)), rel=1e-12)


def test_stroka_ned_plata_hvost_i_flag():
    x, d = dni()
    baza = np.full(len(x), 0.0001)
    r = me.stroka_ned(x, d, None, 0.05, baza)
    xp = x - 0.05 * baza
    assert me.nedelnye_summy(xp, d).sum() == pytest.approx(xp.sum()) and r["n_nedel"] == len(W)
    assert r["msr_dn_opredelen"] == bool(np.isfinite(me.msr(xp)))
    h = me.stroka_ned(x, d, "hud", 0.0, 0.0)               # 10 худших дней вычеркнуты, как в dve_metriki_a.stroka
    keep = np.ones(len(x), bool)
    keep[np.argsort(x)[:10]] = False
    assert h["sharpe_ned"] == pytest.approx(me.metriki_ned(x[keep], d[keep])["sharpe_ned"])
    assert h["sharpe_ned"] != pytest.approx(r["sharpe_ned"])
    assert set(h) == set(me.KOLONKI_NED)
    assert me.stroka_ned(x, d, None, 0.0, 0.0)["msr_dn_opredelen"] is bool(np.isfinite(me.msr(x)))


def test_korotkij_ryad_nan():
    x, d = dni(W[:2])
    assert np.isnan(me.metriki_ned(x, d)["msr_ned"])
