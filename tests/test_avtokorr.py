"""Автокорреляция дневных доходностей (Ж-126): ACF, VR(5), Льюнг–Бокс, серии убытков — синтетика с известным ответом + мутации."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import metriki_e as me

PHI, N = 0.3, 70_000        # 70 000 торговых дней ≈ 269 лет < 292 лет (предел Timedelta в ns)
VR_TEOR = 1 + 2 * sum((1 - k / 5) * PHI**k for k in range(1, 5))        # VR(5) для AR(1): 1 + 2Σ(1−k/5)φ^k


def _daty(n: int) -> np.ndarray:
    return pd.bdate_range("1800-01-06", periods=n).to_numpy().astype("datetime64[D]")      # Пн–Пт, все недели полные


def _ar1(phi: float, n: int = N, seed: int = 1) -> np.ndarray:
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _acf_lag0(r, nlag=5):
    """Мутация: лаг сдвинут — ACF(0)=1 вместо ACF(1)."""
    return np.array([1.0, *me.acf(r, nlag)[:-1]])


def _vr_bez_pyati(x, daty):
    """Мутация: без деления на 5."""
    return me.vr_ned(x, daty)[0] * 5


def test_ar1_acf1_i_vr5():
    x = _ar1(PHI)
    a = me.acf(x, 5)
    assert a[0] == pytest.approx(PHI, abs=0.01) and a[1] == pytest.approx(PHI**2, abs=0.01)
    vr, nw = me.vr_ned(x, _daty(N))
    assert nw == N // 5 and vr == pytest.approx(VR_TEOR, abs=0.03)


def test_belyj_shum_vr_okolo_edinicy():
    x = _ar1(0.0, seed=2)
    assert me.vr_ned(x, _daty(N))[0] == pytest.approx(1.0, abs=0.03)
    assert abs(me.acf(x, 5)).max() < 0.01


def test_mutaciya_lag_sdvinut_lovitsya():
    x = _ar1(PHI)
    assert abs(_acf_lag0(x)[0] - PHI) > 0.5            # мутация даёт 1, а не 0,3
    assert abs(me.acf(x, 5)[0] - PHI) < 0.01          # настоящая функция проходит ту же проверку


def test_mutaciya_vr_bez_deleniya_na_5_lovitsya():
    x, d = _ar1(PHI), _daty(N)
    assert abs(_vr_bez_pyati(x, d) - VR_TEOR) > 3
    assert abs(me.vr_ned(x, d)[0] - VR_TEOR) < 0.03


def test_vr_znak_perevernut_ne_prohodit():
    """Отрицательная автокорреляция: VR < 1 (перепутанный знак ловится)."""
    assert me.vr_ned(_ar1(-PHI), _daty(N))[0] < 1 - 0.1


def test_vr_nepolnye_nedeli_ruchnoj_primer():
    """Две полные недели и одна из 2 дней (праздничная): все недели — 3 суммы, только полные — 2."""
    d = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                        "2024-01-15", "2024-01-16"]).to_numpy().astype("datetime64[D]")
    x = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 5, 5], float)
    vr_v, n_v = me.vr_ned(x, d)
    vr_p, n_p = me.vr_ned(x, d, tolko_polnye=True)
    assert (n_v, n_p) == (3, 2)
    assert vr_v == pytest.approx(np.var([5, 10, 10], ddof=1) / (5 * np.var(x, ddof=1)))
    assert vr_p == pytest.approx(np.var([5, 10], ddof=1) / (5 * np.var(x, ddof=1)))


def test_lyung_boks_sovpadaet_so_statsmodels():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = _ar1(PHI, 3000)
    q, p = me.lyung_boks(x, 5)
    ref = acorr_ljungbox(x, lags=[5])
    assert q == pytest.approx(float(ref.lb_stat.iloc[0]), rel=1e-9) and p == pytest.approx(float(ref.lb_pvalue.iloc[0]), rel=1e-6)
    assert p < 1e-6 and me.lyung_boks(_ar1(0.0, 3000, seed=5), 5)[1] > 0.001


def test_serii_ubytkov_ruchnoj_primer():
    r = np.array([-1, -1, 1, -1, 1, 1, -1, -1, -1, 1], float)
    s = me.serii_ubytkov(r)
    assert s["p_ub"] == pytest.approx(0.6) and s["p_ub_ub"] == pytest.approx(0.5)
    assert s["dlina"] == pytest.approx(2.0) and s["dlina_nezav"] == pytest.approx(2.5)


def test_serii_nezavisimye_i_markov():
    """Независимые знаки: P(у|у) ≈ P(у), длина ≈ 1/(1−p). Серии (Марков, остаётся в убытке с 0,7): P(у|у) ≈ 0,7, длина ≈ 1/0,3."""
    rng = np.random.default_rng(3)
    s0 = me.serii_ubytkov(rng.standard_normal(N))
    assert s0["p_ub_ub"] == pytest.approx(s0["p_ub"], abs=0.01) and s0["dlina"] == pytest.approx(s0["dlina_nezav"], abs=0.03)
    u, L = rng.random(N), np.zeros(N, bool)
    for t in range(1, N):
        L[t] = u[t] < (0.7 if L[t - 1] else 0.3)      # P(у|у) = 0,7, P(у|не у) = 0,3 → стационарная p = 0,5
    s1 = me.serii_ubytkov(np.where(L, -1.0, 1.0))
    assert s1["p_ub_ub"] == pytest.approx(0.7, abs=0.01) and s1["dlina"] == pytest.approx(1 / 0.3, abs=0.1) and s1["dlina"] > s1["dlina_nezav"] + 1


def test_stroka_sharpe_kak_v_tablicah():
    x, d = _ar1(PHI, 5000), _daty(5000)
    r = me.avtokorr_stroka(x, d)
    assert r["sharpe_dn"] == pytest.approx(me.sr_dnevnoy(x) * np.sqrt(252), abs=1e-12)
    assert r["sharpe_ned"] == pytest.approx(me.metriki_ned(x, d)["sharpe_ned"], abs=1e-12)
    assert r["raznost"] == pytest.approx(r["sharpe_ned"] - r["sharpe_dn"]) and r["n_nedel"] == 1000


def _skript():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "scripts" / "analysis"), str(root / "scripts" / "revers")]
    import r1_a_avtokorr as S
    return S


def test_skript_obraztsy_i_sverka_lovit_podmenu():
    """Образцы читаются из опубликованных таблиц; строка с точными значениями проходит сверку, с подменой Sharpe нед. — падает (гейт 1e-12)."""
    S = _skript()
    ob = S.obraztsy()
    sh, sn = ob[("yakor", 0.05, "полные")]
    assert sn < sh < 0 and ("A", 0.0, "2022") in ob and ("A", 0.05, "полные") in ob
    df = pd.DataFrame([dict(kod="yakor", plata_zaem=0.05, period="полные", sharpe_dn=sh, sharpe_ned=sn)])
    assert S.sverka(df, ob)[0] == 0.0
    df["sharpe_ned"] = sn + 1e-6
    assert S.sverka(df, ob)[0] > S.TOL
    df["period"] = "нет такого"
    with pytest.raises(SystemExit):
        S.sverka(df, ob)


def test_skript_stroki_period_i_sharpe_kak_v_metriki():
    S = _skript()
    n = 3 * 260
    d = pd.bdate_range("2022-01-03", periods=n).to_numpy().astype("datetime64[D]")
    x = np.random.default_rng(4).standard_normal(n)
    rows = S.stroki("A", x, d, 0.0)
    assert [r["period"] for r in rows] == ["полные", "2022", "2023", "2024"] and rows[0]["dni"] == n
    assert rows[1]["dni"] + rows[2]["dni"] + rows[3]["dni"] == n
    assert rows[0]["sharpe_ned"] == pytest.approx(me.metriki_ned(x, d)["sharpe_ned"], abs=1e-12)
