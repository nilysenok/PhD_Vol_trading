"""Ж-214, лист 3, V8: альтернативные ВХОДНЫЕ ряды RV (change_type = input_measurement; цель — каноническая RV, У18–У20 GATES_V2) и версия цели Z6 (target_definition).
Z1 `sa_exp_input`, Z2 `sa_roll250_input`: RV_sa(d) = Σ_k r²_k / s_k(d), s_k(d) — медиана ДОЛЕЙ слота r²_k/RV по дням s < d (расширяющаяся / последние 250), нормировка на среднее по слотам
(>= 60 дней, >= 20 наблюдений слота). Z4 `sub20_input`: RV на 20-минутной сетке, среднее двух сдвигов. Z5 `trim_first_input`: RV без первого бара (слот 0).
Профили и ряды строятся из матриц слотов (дни × 53) — функции принимают матрицы, поэтому главный тест обрезки порчу будущих строк проверяет прямо на них."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_dec as DC, hv2_qlog as QG, hv2_seas as Z

MIN_DAYS, MIN_CNT, ROLL = 60, 20, 250
KEYS = {"Z1": "z1", "Z2": "z2", "Z4": "z4", "Z5": "z5"}
KIND = {"Z1": "sa_exp_input", "Z2": "sa_roll250_input", "Z4": "sub20_input", "Z5": "trim_first_input"}


def median_profile(M2: np.ndarray, win: int | None = None, lag: int = 1) -> np.ndarray:
    """(дни × слоты): s_k(d) по дням s <= d − lag (lag = 1 — строго до d); win — последние win дней (None — расширяющаяся). lag = 0 — МУТАНТ (день d в профиле)."""
    n = len(M2)
    tot = np.nansum(M2, axis=1)
    share = np.where(tot[:, None] > 0, M2 / np.where(tot > 0, tot, 1.0)[:, None], np.nan)
    out = np.full(M2.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in range(MIN_DAYS + lag - 1, n):
            lo = max(0, d - lag + 1 - win) if win else 0
            blk = share[lo:d - lag + 1]
            if len(blk) < MIN_DAYS:
                continue
            prof = np.nanmedian(blk, axis=0)
            prof[np.isfinite(blk).sum(axis=0) < MIN_CNT] = np.nan
            out[d] = prof / np.nanmean(prof)
    return out


def rv20(R: np.ndarray) -> np.ndarray:
    """RV (%²) на 20-минутной сетке: доходность = r_k + r_{k+1} (оба слота есть); сетки пар (0,1),(2,3)…(50,51) и (1,2)…(51,52); среднее двух сумм квадратов; нет ни одной пары — NaN."""
    res = []
    for off in (0, 1):
        a = np.arange(off, 52, 2)
        s = R[:, a] + R[:, a + 1]
        res.append(np.where(np.isfinite(s).any(axis=1), np.nansum(s * s, axis=1), np.nan))
    return 0.5 * (res[0] + res[1])


def build_inputs(M2: np.ndarray, R: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Ряды Z1, Z2, Z4, Z5 (%²) на днях index. M2 — r² (%²) по слотам, R — r (%) по слотам."""
    out = pd.DataFrame(index=index)
    out["z1"] = Z.rv_sa(M2, median_profile(M2))
    out["z2"] = Z.rv_sa(M2, median_profile(M2, ROLL))
    out["z4"] = rv20(R)
    out["z5"] = np.where(np.isfinite(M2[:, 1:]).any(axis=1), np.nansum(M2[:, 1:], axis=1), np.nan)
    return out


def with_inputs(w: DC.DWorld, z: pd.DataFrame) -> DC.DWorld:
    comp = {**w.comp, **{k: z[k].reindex(pd.DatetimeIndex(w.p.dates)).to_numpy(float) for k in z.columns}}
    return dataclasses.replace(w, comp=comp)


def features(w: DC.DWorld, key: str) -> np.ndarray:
    return np.column_stack([DC.own(w.comp[key], w.p.seg), w.p.X[:, 3:4]])


def run_input(w: DC.DWorld, key: str, a0: np.ndarray | None = None) -> tuple:
    """Z1/Z2/Z4/Z5: ln RV_{t+1} (каноническая) на [ln X_d, ln X_w, ln X_m, ln(1+J_d)]; -> (σ̂² с подменой на A0, число fallback)."""
    s2 = DC.forecast(w, features(w, key), w.p.lnrv)
    fb = ~np.isfinite(s2)
    a0 = DC.a0_forecast(w) if a0 is None else a0
    return np.where(fb, a0, s2), int(fb.sum())


def run_target(w: DC.DWorld, key: str, a0: np.ndarray | None = None) -> tuple:
    """Z6 (другая цель): модель A0-спецификации на ln RV_sa_{t+1}; σ̂²_RV = c_d·σ̂²_sa, c_d — среднее RV/σ̂²_sa по закрытым (>= 252, иначе без шкалы).
    -> (σ̂² на исходной RV, σ̂²_sa, число fallback). Сравнение — только на исходной RV."""
    sa = DC.forecast(w, features(w, key), DC.lnx(w.comp[key]))
    fb = ~np.isfinite(sa)
    a0 = DC.a0_forecast(w) if a0 is None else a0
    sa = np.where(fb, a0, sa)
    return QG.scale_c(sa, sa, w.target, w.d, w.td), sa, int(fb.sum())


def sigma(w: DC.DWorld, key: str) -> np.ndarray:
    return run_input(w, key)[0]
