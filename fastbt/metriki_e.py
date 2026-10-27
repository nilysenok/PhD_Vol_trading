"""Метрики §6 постановки семьи E (docs/zadaniya/2026-10-27_semya_E.md) над дневным рядом r
избыточной доходности. Везде ddof=1 (std) и √252 (годовая шкала), кроме S и K.

S и K — СМЕЩЁННЫЕ (моментные) оценки: S = m3/m2^1.5, K_изб = m4/m2² − 3, моменты по 1/T.
Это оценщик Бейли–Лопеса де Прадо и Фавра–Галеано; K всегда избыточный (нормаль = 0),
в PSR/DSR внутри формулы подставляется сырой эксцесс K_изб + 3 (в (K−1)/4 у Бейли — γ4, нормаль = 3).

Размерности: SR в PSR и DSR — ДНЕВНОЙ (mean/sd, без √252); SR* в DSR — тоже дневной,
V — дисперсия дневных SR точек сетки. Годовой SR (s9.sharpe) сюда не подаётся.
"""
from __future__ import annotations

from math import e, sqrt
from statistics import NormalDist

import numpy as np

from fastbt.s9 import sharpe  # SR годовой = mean/sd(ddof=1)·√252

__all__ = ["sharpe", "sr_dnevnoy", "asimmetriya", "ekscess", "msr", "sortino_e", "kapital",
           "mdd", "kalmar", "omega", "fko_dolya", "psr", "sr_zvezda", "dsr", "dsr_iz_setki"]

GAMMA = 0.5772
_N = NormalDist()
Z95 = -1.645


def _x(r) -> np.ndarray:
    return np.asarray(r, float)


def sr_dnevnoy(r) -> float:
    """SR в дневной шкале: mean/sd (ddof=1); при нулевом sd — 0."""
    x = _x(r)
    s = np.std(x, ddof=1)
    return float(np.mean(x) / s) if s > 1e-12 else 0.0


def asimmetriya(r) -> float:
    """S — смещённая асимметрия m3/m2^1.5."""
    x = _x(r)
    d = x - x.mean()
    m2 = np.mean(d**2)
    return float(np.mean(d**3) / m2**1.5) if m2 > 1e-24 else 0.0


def ekscess(r) -> float:
    """K — смещённый ИЗБЫТОЧНЫЙ эксцесс m4/m2² − 3."""
    x = _x(r)
    d = x - x.mean()
    m2 = np.mean(d**2)
    return float(np.mean(d**4) / m2**2 - 3.0) if m2 > 1e-24 else 0.0


def msr(r) -> float:
    """mSR Фавра–Галеано: mean·252 / mVaR₉₅, mVaR₉₅ = −(mean + z_cf·sd)·√252. nan при mVaR ≤ 0."""
    x = _x(r)
    m, sd, s, k, z = x.mean(), np.std(x, ddof=1), asimmetriya(x), ekscess(x), Z95
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    mvar = -(m + zcf * sd) * sqrt(252)
    return float(m * 252 / mvar) if mvar > 1e-12 else float("nan")


def sortino_e(r) -> float:
    """Сортино по §6: mean·252 / (sd отрицательных r, ddof=1 · √252).
    Отличие от msp.metrics: там sd берётся по отрицательным ПОЛНЫМ доходностям, а не по избыточным r."""
    x = _x(r)
    neg = x[x < 0]
    if len(neg) < 2:
        return float("nan")
    d = np.std(neg, ddof=1) * sqrt(252)
    return float(np.mean(x) * 252 / d) if d > 1e-12 else float("nan")


def kapital(r) -> np.ndarray:
    """Капитал на 1: cumprod(1+r), с начальной единицей в позиции 0 (длина T+1)."""
    return np.concatenate([[1.0], np.cumprod(1.0 + _x(r))])


def mdd(r) -> float:
    """Максимальная просадка капитала от пика (доля, ≥ 0)."""
    k = kapital(r)
    return float(np.max(1.0 - k / np.maximum.accumulate(k)))


def kalmar(r) -> float:
    """Кальмар: годовая доходность (mean·252) / MDD капитала; nan при MDD = 0."""
    d = mdd(r)
    return float(np.mean(_x(r)) * 252 / d) if d > 1e-12 else float("nan")


def omega(r) -> float:
    """Omega(0) = Σ r⁺ / Σ|r⁻|; inf, если убытков нет."""
    x = _x(r)
    dn = -x[x < 0].sum()
    return float(x[x > 0].sum() / dn) if dn > 0 else float("inf")


def fko_dolya(r) -> float:
    """FKO по §6: доля дней, когда капитал > (максимум − 10 %), максимум — бегущий пик капитала.
    Не путать с колонкой fko_nad_poroga_pct в shag0.py (там другая база/порог, в процентах)."""
    k = kapital(r)[1:]
    pik = np.maximum.accumulate(np.concatenate([[1.0], k]))[1:]
    return float(np.mean(k > pik * 0.9))


def _znam(sr: float, s: float, k_izb: float) -> float:
    return sqrt(max(1.0 - s * sr + (k_izb + 3.0 - 1.0) / 4.0 * sr**2, 1e-12))


def psr(r, sr_star: float = 0.0) -> float:
    """PSR Бейли–Лопеса де Прадо: Φ[(SR_d − SR*)·√(T−1) / √(1 − S·SR_d + (K_сырой−1)/4·SR_d²)]."""
    x = _x(r)
    sr = sr_dnevnoy(x)
    return float(_N.cdf((sr - sr_star) * sqrt(len(x) - 1) / _znam(sr, asimmetriya(x), ekscess(x))))


def sr_zvezda(n: int, v: float) -> float:
    """SR* = √V·((1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))); N ≤ 1 — выбора нет, SR* = 0."""
    if n <= 1:
        return 0.0
    return float(sqrt(v) * ((1 - GAMMA) * _N.inv_cdf(1 - 1 / n) + GAMMA * _N.inv_cdf(1 - 1 / (n * e))))


def dsr(r, n: int, v: float) -> float:
    """DSR(N, V) = PSR(r; SR*(N, V)); V — дисперсия ДНЕВНЫХ SR точек сетки."""
    return psr(r, sr_zvezda(n, v))


def dsr_iz_setki(sr_vseh_tochek, r_vybrannoy) -> float:
    """DSR по сетке: N = len(sr_vseh_tochek), V = var(sr, ddof=1). SR точек — ДНЕВНЫЕ (sr_dnevnoy)."""
    s = _x(sr_vseh_tochek)
    return dsr(r_vybrannoy, len(s), float(np.var(s, ddof=1)) if len(s) > 1 else 0.0)
