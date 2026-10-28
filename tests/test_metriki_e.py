"""Метрики §6 семьи E (fastbt/metriki_e.py): синтетика, независимая сверка, мутации §8 п. 12.
Мутации — только в памяти (monkeypatch и эталон с подменой)."""
from __future__ import annotations

import sys
from math import e, erf, sqrt
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastbt import metriki_e as M  # noqa: E402
from fastbt.s9 import sharpe as s9_sharpe  # noqa: E402

G = 0.5772


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def Phi_inv(p):  # бисекция: независимо от statistics.NormalDist
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def norm_series(seed, n=2000, mu=0.0005, sd=0.01):
    return np.random.default_rng(seed).normal(mu, sd, n)


def skew_series(seed=3, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.003, 0.006, n)
    x[rng.random(n) < 0.03] -= 0.05
    return x


def ref_moments(x, raw=False):
    d = x - x.mean()
    m2, m3, m4 = (d**2).mean(), (d**3).mean(), (d**4).mean()
    return m3 / m2**1.5, m4 / m2**2 - (0 if raw else 3)


def ref_msr(x, z=-1.645, raw_k=False, ddof=1):
    s, k = ref_moments(x, raw_k)
    zc = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    m, sd = x.mean(), x.std(ddof=ddof)
    return m * 252 / (-(m + zc * sd) * sqrt(252))


def ref_psr(x, sr_star=0.0):
    sr = x.mean() / x.std(ddof=1)
    s, g4 = ref_moments(x, raw=True)
    return Phi((sr - sr_star) * sqrt(len(x) - 1) / sqrt(1 - s * sr + (g4 - 1) / 4 * sr**2))


def ref_sr_star(n, v):
    return sqrt(v) * ((1 - G) * Phi_inv(1 - 1 / n) + G * Phi_inv(1 - 1 / (n * e)))


def test_sharpe_est_s9():
    x = norm_series(1)
    assert M.sharpe is s9_sharpe
    assert M.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * sqrt(252))


def test_normalnyj_msr_i_psr():
    x = norm_series(1, n=200000, mu=0.0004, sd=0.01)
    m, sd = x.mean(), x.std(ddof=1)
    ozhid = m * 252 / ((1.645 * sd - m) * sqrt(252))  # z_cf -> z при S=K=0
    assert M.msr(x) == pytest.approx(ozhid, rel=0.01)
    assert M.msr(x) == pytest.approx(M.sharpe(x) / 1.645, rel=0.05)  # mean << 1.645·sd
    y = norm_series(2)
    assert M.psr(y) == pytest.approx(Phi(M.sr_dnevnoy(y) * sqrt(len(y) - 1)), abs=0.02)


def test_otricatelnaya_asimmetriya_msr_nizhe():
    x = skew_series()
    assert M.asimmetriya(x) < -1
    normal_ver = x.mean() * 252 / ((1.645 * x.std(ddof=1) - x.mean()) * sqrt(252))
    assert M.msr(x) < normal_ver
    assert M.msr(x) < M.sharpe(x)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_sverka_psr_dsr_nezavisimo(seed):
    x = norm_series(seed, n=1500) if seed != 13 else skew_series(seed)
    assert M.psr(x) == pytest.approx(ref_psr(x), abs=1e-9)
    n, v = 40, 0.0004
    assert M.sr_zvezda(n, v) == pytest.approx(ref_sr_star(n, v), abs=1e-6)
    assert M.dsr(x, n, v) == pytest.approx(ref_psr(x, ref_sr_star(n, v)), abs=1e-6)
    sr_pts = np.random.default_rng(seed).normal(0.02, 0.02, n)
    ozh = ref_psr(x, ref_sr_star(n, sr_pts.var(ddof=1)))
    assert M.dsr_iz_setki(sr_pts, x) == pytest.approx(ozh, abs=1e-6)


def test_s_k_sverka():
    x = skew_series()
    s, k = ref_moments(x)
    assert M.asimmetriya(x) == pytest.approx(s) and M.ekscess(x) == pytest.approx(k)
    assert abs(M.ekscess(norm_series(5, n=500000))) < 0.05  # нормаль: избыточный ≈ 0


def test_omega_dva_znacheniya():
    assert M.omega([0.02, -0.01] * 5) == pytest.approx(2.0)
    assert M.omega([0.02, -0.01, 0.02]) == pytest.approx(4.0)
    assert M.omega([0.01, 0.02]) == float("inf")


def test_kalmar_i_mdd():
    r = [0.10, -0.20, 0.05]
    assert M.mdd(r) == pytest.approx(0.20)  # 1.1 -> 0.88
    assert M.kalmar(r) == pytest.approx(np.mean(r) * 252 / 0.20)
    assert M.mdd([0.01, 0.02]) == 0.0 and np.isnan(M.kalmar([0.01, 0.02]))
    assert M.mdd([-0.1, -0.1]) == pytest.approx(0.19)


def test_sortino_po_p6_ruchnoj():
    r = [0.02, -0.01, 0.03, -0.03]
    sd_neg = sqrt(((-0.01 + 0.02) ** 2 + (-0.03 + 0.02) ** 2) / 1)  # ddof=1, среднее -0,02
    ozh = 0.0025 * 252 / (sd_neg * sqrt(252))
    assert M.sortino_e(r) == pytest.approx(ozh)
    assert M.sortino_e(r) != pytest.approx(0.0025 * 252 / (np.std([-0.01, -0.03]) * sqrt(252)))  # ddof=0


def test_fko_dolya():
    # 1 -> 0.95 -> 0.855 (-14.5% от пика)-> 0.9405 : дни: 0.95 да, 0.855 нет, 0.9405 да (> 0.9)
    r = [-0.05, -0.10, 0.10]
    assert M.fko_dolya(r) == pytest.approx(2 / 3)
    assert M.fko_dolya([0.01, 0.01]) == 1.0


def test_mutaciya_p12_N_v_edinicu_dsr_dolzhen_vyrasti():
    x = norm_series(21, n=1500, mu=0.0008)
    v = 0.0004
    d_n = M.dsr(x, 100, v)
    d_1 = M.dsr(x, 1, v)
    assert d_1 > d_n and d_1 == pytest.approx(M.psr(x))
    assert M.dsr(x, 10, v) > M.dsr(x, 100, v)  # монотонно по N
    # подмена N на 1 внутри dsr_iz_setki: результат обязан отличаться от настоящего
    pts = np.random.default_rng(0).normal(0.03, 0.02, 100)
    mut = M.dsr(x, 1, float(np.var(pts, ddof=1)))
    assert mut > M.dsr_iz_setki(pts, x) + 1e-3


def test_mutacii_v_pamyati(monkeypatch):
    x = skew_series()
    base = M.msr(x)
    assert base == pytest.approx(ref_msr(x), rel=1e-9)
    # эталон с мутацией расходится: знак z, сырой K, ddof=0
    assert abs(ref_msr(x, z=1.645) - base) > 1e-3
    assert abs(ref_msr(x, raw_k=True) - base) > 1e-3
    assert abs(ref_msr(x, ddof=0) - base) > 1e-6
    # мутации самого модуля ловятся тестом сверки с эталоном
    monkeypatch.setattr(M, "Z95", 1.645)
    assert M.msr(x) != pytest.approx(ref_msr(x), abs=1e-3)  # nan тоже «поймано»
    monkeypatch.setattr(M, "Z95", -1.645)
    monkeypatch.setattr(M, "ekscess", lambda r: ref_moments(np.asarray(r), raw=True)[1])
    assert abs(M.msr(x) - ref_msr(x)) > 1e-3
    monkeypatch.undo()
    assert M.msr(x) == pytest.approx(ref_msr(x), rel=1e-9)
    # ddof=0 в SR: дневной SR отличается от эталона на малой выборке
    y = np.array([0.01, -0.02, 0.015, 0.0, 0.005])
    assert M.sr_dnevnoy(y) == pytest.approx(y.mean() / y.std(ddof=1))
    assert abs(M.sr_dnevnoy(y) - y.mean() / y.std(ddof=0)) > 1e-3
