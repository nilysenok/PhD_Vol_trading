"""Ж-214, этап V: матрицы доходностей по слотам времени суток и причинный (PIT) сезонный профиль.

Слот = (begin − 10:00) / 10 мин, 0..52 (53 слота; в режиме до апреля 2015 слот 52 отсутствует). Профиль дня d — среднее r² слота по ДНЯМ < d
(лаг 1), нормировано на среднее по слотам с данными; RV_sa(d) = Σ r²_слот / s_слот(d). Это ОТДЕЛЬНАЯ гипотеза (другая цель), в каноническую
RV не подставляется; сравнение — на исходной цели (лист 3).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_sched as S

NSLOT = 53
MIN_DAYS, MIN_CNT = 60, 20
LAG = 1


def slot_matrix(r: pd.DataFrame, index: pd.Index, power: int = 2) -> np.ndarray:
    """Строки баров -> (дни × слоты): r^power (power=2 — в %², power=1 — доходность в %); нет бара — NaN."""
    M = np.full((len(index), NSLOT), np.nan)
    pos = index.get_indexer(pd.DatetimeIndex(r["date"]))
    slot = (r["mins"].to_numpy() - S.START_MIN) // S.BAR_MIN
    ok = (pos >= 0) & (slot >= 0) & (slot < NSLOT)
    scale = D.SCALE if power == 2 else 100.0
    M[pos[ok], slot[ok]] = r["r"].to_numpy(float)[ok] ** power * scale
    return M


def pit_profile(M2: np.ndarray, lag: int = LAG) -> np.ndarray:
    """(дни × слоты): s_k(d) по дням <= d − lag; NaN, если < MIN_DAYS дней или < MIN_CNT наблюдений слота. lag=0 — МУТАНТ (включает день d)."""
    fin = np.isfinite(M2)
    cs = np.cumsum(np.where(fin, M2, 0.0), axis=0)
    cn = np.cumsum(fin, axis=0)
    n = len(M2)
    sh = lambda a: np.vstack([np.zeros((lag, a.shape[1])), a[:n - lag]]) if lag else a  # noqa: E731
    ps, pc = sh(cs), sh(cn).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = np.where(pc >= MIN_CNT, ps / pc, np.nan)
        mean = np.nanmean(prof, axis=1, keepdims=True)
        s = prof / mean
    s[np.arange(n) < MIN_DAYS + lag - 1] = np.nan
    return s


def rv_sa(M2: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Σ r²/s по слотам дня; NaN, если у наблюдённого слота нет профиля."""
    obs = np.isfinite(M2)
    bad = (obs & ~np.isfinite(s)).any(axis=1) | ~obs.any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(obs, M2 / s, 0.0).sum(axis=1)
    return np.where(bad, np.nan, v)


def build(r: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    M2 = slot_matrix(r, index)
    s = pit_profile(M2)
    out = pd.DataFrame(index=index)
    out["rv_sa"] = rv_sa(M2, s)
    out["rv_sa_ratio"] = out["rv_sa"] / (np.nansum(M2, axis=1))
    return out
