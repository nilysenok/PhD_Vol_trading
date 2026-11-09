"""V7-1a-E: представление z_d. z = [среднее по барам окна ‖ последнее] скрытое состояние энкодера (8+8 = 16), затем главные компоненты до d_z = 4.
PIT: стандартизация, компоненты и масштаб счётов оцениваются только по z дат < 01.01.Y (энкодер того же года Y); z дня d — только из баров окна ≤ d."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s14 import v7_1e_net as NET
from models_v2.s14.v7_1e_pre import deterministic

D_Z = 4


def encode(model, X, M) -> np.ndarray:
    """X (N, W, 53, 11), M (N, W, 53) -> (N, 16): среднее и последнее скрытое состояние по барам окна (слоты без бара не входят)."""
    x, m = NET.to_seq(X, M), torch.as_tensor(M).reshape(len(M), -1)
    with deterministic(), torch.no_grad():
        h = model.hidden(x)                                   # (N, 8, L)
    mf = m.to(h.dtype)
    cnt = mf.sum(1, keepdim=True).clamp(min=1.0)
    mean = (h * mf[:, None, :]).sum(2) / cnt
    last_i = (m.shape[1] - 1 - torch.flip(m, [1]).to(torch.int64).argmax(1))
    last = h[torch.arange(len(h)), :, last_i] * m.any(1, keepdim=True).to(h.dtype)
    return torch.cat([mean, last], 1).numpy().astype(np.float64)


def _past_mask(dates, year: int) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(dates) < pd.Timestamp(year, 1, 1))


def fit_pca_past(z: np.ndarray, dates, year: int, d_z: int = D_Z) -> dict:
    """Стандартизация и главные компоненты по z дат < 01.01.year. Знак компоненты: наибольшая по модулю нагрузка > 0."""
    zp = z[_past_mask(dates, year)]
    if len(zp) <= d_z:
        raise ValueError("мало прошлых z для главных компонент")
    mu, sd = zp.mean(0), zp.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    comps = np.linalg.svd((zp - mu) / sd, full_matrices=False)[2][:d_z]
    comps = comps * np.sign(comps[np.arange(d_z), np.abs(comps).argmax(1)])[:, None]
    ssd = np.einsum("ij,kj->ik", (zp - mu) / sd, comps).std(0)
    return dict(mu=mu, sd=sd, comps=comps, ssd=np.where(ssd > 0, ssd, 1.0), year=year, n_past=len(zp))


def transform_pca(p: dict, z: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ik", (z - p["mu"]) / p["sd"], p["comps"]) / p["ssd"]


def z_pit(model, pack: dict, rows: np.ndarray, year: int, d_z: int = D_Z) -> tuple:
    """z (len(rows), d_z) для дней `rows` энкодером года `year`; компоненты — по z дат < 01.01.year. -> (z, pca)."""
    raw = encode(model, pack["X"][rows], pack["mask"][rows])
    p = fit_pca_past(raw, np.asarray(pack["days"])[rows], year, d_z)
    return transform_pca(p, raw), p
