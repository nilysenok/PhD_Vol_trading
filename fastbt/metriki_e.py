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
import pandas as pd

from fastbt.s9 import sharpe  # SR годовой = mean/sd(ddof=1)·√252

__all__ = ["sharpe", "sr_dnevnoy", "asimmetriya", "ekscess", "msr", "sortino_e", "kapital",
           "nedelnye_summy", "metriki_ned", "stroka_ned", "acf", "vr_ned", "lyung_boks", "serii_ubytkov", "avtokorr_stroka","mdd", "kalmar", "omega", "fko_dolya", "psr", "sr_zvezda", "dsr", "dsr_iz_setki"]

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


def msr(r, g: int = 252) -> float:
    """mSR Фавра–Галеано: mean·g / mVaR₉₅, mVaR₉₅ = −(mean + z_cf·sd)·√g. nan при mVaR ≤ 0.
    g — число периодов в году: 252 (дневной ряд, по умолчанию) или 52 (недельный, Ж-124 п. 3)."""
    x = _x(r)
    m, sd, s, k, z = x.mean(), np.std(x, ddof=1), asimmetriya(x), ekscess(x), Z95
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    mvar = -(m + zcf * sd) * sqrt(g)
    return float(m * g / mvar) if mvar > 1e-12 else float("nan")


def sortino_e(r, g: int = 252) -> float:
    """Сортино по §6: mean·g / (sd отрицательных r, ddof=1 · √g); g = 252 (дневной) или 52 (недельный).
    Отличие от msp.metrics: там sd берётся по отрицательным ПОЛНЫМ доходностям, а не по избыточным r."""
    x = _x(r)
    neg = x[x < 0]
    if len(neg) < 2:
        return float("nan")
    d = np.std(neg, ddof=1) * sqrt(g)
    return float(np.mean(x) * g / d) if d > 1e-12 else float("nan")


NED = 52        # недель в году: годовая шкала недельных метрик (Ж-124 п. 3)
KOLONKI_NED = ("sharpe_ned", "sortino_ned", "msr_ned", "skew_ned", "kurt_ned", "n_nedel", "msr_dn_opredelen")


def nedelnye_summy(r, daty) -> np.ndarray:
    """Недельная избыточная доходность = сумма дневных за торговые дни недели W-FRI (неполные края входят, пустых недель нет)."""
    idx = pd.DatetimeIndex(np.asarray(daty).astype("datetime64[D]"))
    s = pd.Series(_x(r), index=idx).sort_index()
    return s.groupby(s.index.to_period("W-FRI")).sum().to_numpy()


def metriki_ned(r, daty) -> dict:
    """Sharpe, Sortino, mSR (√52, правило области как у дневного: вне области nan), skew, kurt и число недель по недельным суммам."""
    w = nedelnye_summy(r, daty)
    if len(w) < 3:
        return dict(sharpe_ned=np.nan, sortino_ned=np.nan, msr_ned=np.nan, skew_ned=np.nan, kurt_ned=np.nan, n_nedel=len(w))
    return dict(sharpe_ned=float(sr_dnevnoy(w) * sqrt(NED)), sortino_ned=sortino_e(w, NED), msr_ned=msr(w, NED),
                skew_ned=asimmetriya(w), kurt_ned=ekscess(w), n_nedel=len(w))


def stroka_ned(x, daty, hvost: str | None = None, plata: float = 0.0, baza=0.0) -> dict:
    """Недельные колонки строки таблицы; отбор дней — как в scripts/analysis/dve_metriki_a.stroka (плата за заём, hvost 'hud'/'luch').
    msr_dn_opredelen — определён ли дневной mSR на тех же днях (True/False)."""
    x = _x(x) - plata * _x(baza)
    keep = np.ones(len(x), bool)
    if hvost:
        keep[np.argsort(x)[:10] if hvost == "hud" else np.argsort(x)[-10:]] = False
    xs = x[keep]
    return dict(**metriki_ned(xs, np.asarray(daty)[keep]), msr_dn_opredelen=bool(np.isfinite(msr(xs))))


def acf(r, nlag: int = 5) -> np.ndarray:
    """Выборочная ACF лагов 1..nlag: r_k = Σ_{t≥k}(x_t−m)(x_{t−k}−m) / Σ(x_t−m)² (одна формула; та же, что в Льюнге–Боксе и statsmodels.acf)."""
    d = _x(r) - np.mean(_x(r))
    den = float(np.sum(d * d))
    return np.array([float(np.sum(d[k:] * d[:-k])) / den if den > 1e-24 else np.nan for k in range(1, nlag + 1)])


def vr_ned(r, daty, tolko_polnye: bool = False) -> tuple[float, int]:
    """VR(5) = Var(недельные суммы W-FRI, ddof=1) / (5 · Var(дневной, ddof=1)); VR > 1 — положительная автокорреляция (недельный Шарп ниже дневного).
    По умолчанию недели те же, что в Sharpe нед. (nedelnye_summy: неполные края и праздничные недели входят как есть);
    tolko_polnye=True — только недели ровно из 5 торговых дней. Возвращает (VR, число недель)."""
    s = pd.Series(_x(r), index=pd.DatetimeIndex(np.asarray(daty).astype("datetime64[D]"))).sort_index()
    g = s.groupby(s.index.to_period("W-FRI"))
    w = g.sum().to_numpy()[(g.count() == 5).to_numpy()] if tolko_polnye else g.sum().to_numpy()
    if len(w) < 2 or np.var(_x(r), ddof=1) < 1e-24:
        return float("nan"), len(w)
    return float(np.var(w, ddof=1) / (5 * np.var(_x(r), ddof=1))), len(w)


def lyung_boks(r, m: int = 5) -> tuple[float, float]:
    """Q(m) = T(T+2)·Σ r_k²/(T−k), p = P(χ²_m > Q)."""
    from scipy.stats import chi2
    T, a = len(r), acf(r, m)
    q = float(T * (T + 2) * np.sum(a**2 / (T - np.arange(1, m + 1))))
    return q, float(chi2.sf(q, m))


def serii_ubytkov(r) -> dict:
    """Убыток = r < 0. p_ub — доля убыточных дней; p_ub_ub = P(убыток d+1 | убыток d) по соседним торговым дням;
    dlina — средняя длина серии убыточных дней подряд; dlina_nezav = 1/(1−p_ub) — ожидаемая при независимости (геометрическое)."""
    L = _x(r) < 0
    p, n = float(L.mean()), int(L[:-1].sum())
    runs = int(L[0]) + int((L[1:] & ~L[:-1]).sum())
    return dict(p_ub=p, p_ub_ub=float((L[:-1] & L[1:]).sum() / n) if n else float("nan"),
                dlina=float(L.sum() / runs) if runs else float("nan"), dlina_nezav=1 / (1 - p) if p < 1 else float("nan"))


def avtokorr_stroka(x, daty) -> dict:
    """Строка «автокорреляция дневных доходностей» (Ж-126): Sharpe дн./нед. (как в таблицах), ACF 1–5, ΣACF(1..4), VR(5), Q(5), серии."""
    x, a = _x(x), acf(x, 5)
    sd, sn = float(sr_dnevnoy(x) * sqrt(252)), metriki_ned(x, daty)["sharpe_ned"]
    q, p = lyung_boks(x, 5)
    vr, nw = vr_ned(x, daty)
    vrp, nwp = vr_ned(x, daty, tolko_polnye=True)
    return dict(sharpe_dn=sd, sharpe_ned=sn, raznost=sn - sd, **{f"acf{k}": float(a[k - 1]) for k in range(1, 6)}, summa_acf14=float(a[:4].sum()),
                vr5=vr, n_nedel=nw, vr5_polnye=vrp, n_polnyh_nedel=nwp, q5=q, p_q5=p, dni=len(x), **serii_ubytkov(x))


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
