"""Фильтры оси `filtr` по собственным рядам тикера (контракт etap1 §2.1; Л-4, прогон этапа 1).

Только данные ≤ close d; значение — bool «вход разрешён» по дням. NaN в условии → False (нет истории — запрет).
Ряды-«сырьё» для Д1 открыты: z_series, rg_series, v_series, u_series.
sd — сторона сигнала дня (+1, −1, 0); от неё зависит только post_shok (и rel_sila_imoex в _rynok).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prev(x: np.ndarray, k: int = 1) -> np.ndarray:
    """x_{d−k}; первые k — NaN."""
    x = np.asarray(x, float)
    return np.r_[np.full(k, np.nan), x[:-k]] if k else x


def ret(close: np.ndarray) -> np.ndarray:
    """r_d = ln(c_d / c_{d−1}); r_0 = NaN."""
    c = np.asarray(close, float)
    r = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(c[1:] / c[:-1])
    return r


def z_series(r: np.ndarray, sig22: np.ndarray) -> np.ndarray:
    """z_d = r_d / σ̂_{d−1} (σ̂ — rv22, известна на 18:50 d−1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(r, float) / prev(sig22)


def win_any(f: np.ndarray, w: int) -> np.ndarray:
    """Есть True в [d−w+1; d]."""
    c = np.cumsum(np.asarray(f, bool).astype(int))
    return (c - np.r_[np.zeros(w, int), c[:-w]]) > 0


def post_shok(z: np.ndarray, Z: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """В [d−4; d] есть день с −sd·z ≥ 3 и |Z| ≤ 1 (Т16); сторона — sd_d."""
    with np.errstate(invalid="ignore"):
        calm = np.abs(Z) <= 1
        lng = win_any((-z >= 3) & calm, 5)       # шок вниз — для лонга
        sht = win_any((z >= 3) & calm, 5)
    sd = np.asarray(sd, float)
    return np.where(sd > 0, lng, np.where(sd < 0, sht, False))


def rg_series(high, low, close, n: int = 20) -> np.ndarray:
    """Rg_d = (max high − min low за n дней [d−n+1; d]) / c_d."""
    h = pd.Series(np.asarray(high, float)).rolling(n, min_periods=n).max().to_numpy()
    lo = pd.Series(np.asarray(low, float)).rolling(n, min_periods=n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (h - lo) / np.asarray(close, float)


def szhatie(rg: np.ndarray) -> np.ndarray:
    """Rg_d ≤ min Rg за [d−119; d] + 1e-12 (Т17); окно из 120 значений обязательно."""
    m = pd.Series(rg).rolling(120, min_periods=120).min().to_numpy()
    with np.errstate(invalid="ignore"):
        return rg <= m + 1e-12


def u_series(rv5: np.ndarray, rv66: np.ndarray) -> np.ndarray:
    """U_d = (σ_rv5 / σ_rv66)²."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(rv5, float) / np.asarray(rv66, float)) ** 2


def rv_perevorot(u: np.ndarray, rv5: np.ndarray) -> np.ndarray:
    """В [d−10; d−1] был день с U > 3, и σ_rv5,d < σ_rv5,d−1 (Т19)."""
    with np.errstate(invalid="ignore"):
        was = prev(win_any(np.nan_to_num(u, nan=0.0) > 3, 10).astype(float)) > 0.5
        return was & (np.asarray(rv5, float) < prev(rv5))


def v_series(vol: np.ndarray) -> np.ndarray:
    """V_d = vol_d / mean(vol_{d−20…d−1}); при нулевом среднем — NaN."""
    m = pd.Series(np.asarray(vol, float)).rolling(20, min_periods=20).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, np.asarray(vol, float) / m, np.nan)


def obyom(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """V > 3 и |z_d| < 0,5 (Т20)."""
    with np.errstate(invalid="ignore"):
        return (v > 3) & (np.abs(z) < 0.5)
