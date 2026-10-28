"""Уровни 1–2 Ж-095 для S2 (контртренд, Bollinger) и S3 (тренд, Donchian): скор, контекст g, размер, порог +1R.

Чистые функции на numpy, без движка. Массивы [N, T]: N тикеров (пул Sk), T торговых дней; значение в колонке d известно к 18:50 d.
Исполнение (open d+1, комиссия, плата за шорт, ставка) — не здесь: движок exec_e* / EvaluatorV2 принимает s_itog и размер как вес w[d].
Определения — docs/resheniya/resheniya.md Ж-095 пп. 1.1–1.4, 2.2; центры — fastbt/designs/sweep/zh047_S{2,3}_v3.yaml (ветка arh-setki-v2).
❓ Умолчания: ранг пулом по тикерам (min_nabl 20 мало для одного тикера); объём SMA20 — по d−20…d−1; «согласие» S2 — относительный z того же знака, что z.
"""
from __future__ import annotations

import numpy as np

OKNO_RANG, MIN_NABL, KAPITAL, DOLYA_OBOROTA = 252, 20, 200e6, 0.05
POROG_C, RHO_C, KS_C = 0.5, 0.003, 2.0                      # центр: порог 0,5, ρ = 0,3 %, k_s = 2σ


def rang_pul(m: np.ndarray, okno: int = OKNO_RANG, min_nabl: int = MIN_NABL) -> np.ndarray:
    """Нормировка (Ж-095 п. 1.2): доля ненулевых величин пула [тикеры × окно d−okno+1…d], не больше m[i,d]. Нет события (m = 0/nan) или < min_nabl → 0."""
    m = np.asarray(m, float)
    out = np.zeros_like(m)
    for d in range(m.shape[1]):
        w = m[:, max(0, d - okno + 1):d + 1]
        v = w[np.isfinite(w) & (w > 0)]
        if len(v) < min_nabl:
            continue
        x = m[:, d]
        ok = np.isfinite(x) & (x > 0)
        out[ok, d] = (v[None, :] <= x[ok, None]).mean(axis=1)
    return out


def _skol(a: np.ndarray, n: int, f) -> np.ndarray:
    """f по скользящему окну из n значений, оканчивающемуся в d (строки — тикеры); первые n−1 дней — nan."""
    out = np.full(a.shape, np.nan)
    for d in range(n - 1, a.shape[1]):
        out[:, d] = f(a[:, d - n + 1:d + 1])
    return out


def z_boll(close: np.ndarray, okno: int = 20) -> np.ndarray:
    """z = (close − SMA)/STD по окну, включающему d (ddof = 0, как полосы Боллинджера)."""
    sma, sd = _skol(close, okno, lambda w: w.mean(axis=1)), _skol(close, okno, lambda w: w.std(axis=1))
    return np.where(sd > 0, (close - sma) / np.where(sd > 0, sd, 1.0), np.nan)


def skor_kontrtrend(close, okno: int = 20, z0: float = 2.0, obrezka: float = 3.0, norma: bool = True) -> np.ndarray:
    """S2: сторона против отклонения (z < −z0 → +1), модуль = ранг величины min(|z|, 3) − z0 (> 0 только при |z| > z0). norma=False — без ранга (для мутации)."""
    z = z_boll(np.asarray(close, float), okno)
    m = np.where(np.abs(z) > z0, np.minimum(np.abs(z), obrezka) - z0, 0.0)
    m = np.nan_to_num(m)
    return -np.sign(np.nan_to_num(z)) * (rang_pul(m) if norma else np.minimum(m / (obrezka - z0), 1.0))


def skor_trend(close, high, low, dc_okno: int = 20, er_okno: int = 10, norma: bool = True) -> np.ndarray:
    """S3: сторона пробоя close d за max(high)/min(low) окна d−w…d−1; модуль = ранг efficiency ratio (|c_d − c_{d−n}|/Σ|Δc| за n дней)."""
    c, h, l = (np.asarray(x, float) for x in (close, high, low))
    hi = _skol(h, dc_okno, lambda w: w.max(axis=1))
    lo = _skol(l, dc_okno, lambda w: w.min(axis=1))
    hi, lo = np.roll(hi, 1, axis=1), np.roll(lo, 1, axis=1)          # окно до d−1
    hi[:, :dc_okno], lo[:, :dc_okno] = np.nan, np.nan
    storona = np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0))
    dc = np.abs(np.diff(c, axis=1, prepend=np.nan))
    put = _skol(dc, er_okno + 1, lambda w: w[:, 1:].sum(axis=1))
    chg = np.full(c.shape, np.nan)
    chg[:, er_okno:] = np.abs(c[:, er_okno:] - c[:, :-er_okno])
    er = np.nan_to_num(np.where(put > 0, chg / np.where(put > 0, put, 1.0), 0.0))
    m = np.where(storona != 0, er, 0.0)
    return storona * (rang_pul(m) if norma else np.minimum(m, 1.0))


def g_ne_novost(r, sigma, obyom, k_r: float = 2.5, k_v: float = 2.0, okno: int = 20) -> np.ndarray:
    """|r_d| < 2,5σ_d и объём_d < 2·SMA20 (среднее объёма d−20…d−1, без самого дня)."""
    v = np.asarray(obyom, float)
    sma = np.full(v.shape, np.nan)
    sma[:, okno:] = _skol(v, okno, lambda w: w.mean(axis=1))[:, okno - 1:-1]
    return ((np.abs(r) < k_r * sigma) & (v < k_v * sma)).astype(float)


def g_soglasie_s2(z, z_rynok) -> np.ndarray:
    """Относительный z к IMOEX (z_i − z_IMOEX, оба на d) того же знака, что z_i: отклонение не целиком рыночное. z_rynok — длины T."""
    zr = z - np.asarray(z_rynok, float)[None, :]
    return (np.sign(zr) == np.sign(z)).astype(float) * np.isfinite(zr)


def g_soglasie_s3(storona, imoex_close, okno: int = 50) -> np.ndarray:
    """Сторона пробоя совпадает со стороной IMOEX к MA50 (close_d против среднего d−49…d)."""
    m = np.asarray(imoex_close, float)
    ma = _skol(m[None, :], okno, lambda w: w.mean(axis=1))[0]
    return (np.sign(storona) == np.sign(m - ma)[None, :]).astype(float) * np.isfinite(ma)[None, :]


def g_likvidnost(razmer_rub, oborot_rub, dolya: float = DOLYA_OBOROTA, okno: int = 20) -> np.ndarray:
    """Позиция ≤ 5 % медианного оборота за okno дней, оканчивающихся в d (оборот — из 1-мин. баров, руб.)."""
    med = _skol(np.asarray(oborot_rub, float), okno, lambda w: np.median(w, axis=1))
    return (razmer_rub <= dolya * med).astype(float)


def razmer_rub(sigma, s_itog, rho: float = RHO_C, k_s: float = KS_C, kapital: float = KAPITAL) -> np.ndarray:
    """Ж-095 п. 2.2: ρ·капитал/(k_s·σ_d)·|s_итог|, руб. Тот же σ_d, что у стопа на день решения d (не d+1)."""
    return rho * kapital / (k_s * np.asarray(sigma, float)) * np.abs(s_itog)


def vhod_maska(s_itog, porog: float = POROG_C) -> np.ndarray:
    """Вход при |s_итог| ≥ порога; знак — сторона."""
    return np.where(np.abs(s_itog) >= porog, np.sign(s_itog), 0.0)


def uroven_1R(storona: int, vhod_px: float, sigma, d_sig: int, k_s: float = KS_C) -> float:
    """Цена +1R: 1R = k_s·σ[d_sig]·цена входа, d_sig — день решения (σ известна к 18:50 d, не d+1); на ней снимается ½ позиции (S3)."""
    return vhod_px * (1.0 + storona * k_s * float(sigma[d_sig]))
