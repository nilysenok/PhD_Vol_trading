"""V7-4a, G46: дрейф энкодера (диагностика, Ж-235 п. 5; частота переобучения не меняется, в прогнозы ничего не возвращается). Для каждого года Y = 2019…2026 и seed 0–4:
 • потеря задачи-предлога (MSE на ln(r²_{j+1}+ε), `v7_1e_pre`, ε — Stage A года Y) у энкодера Stage B (`4/encB`) и у энкодера Stage A (`stageA`, веса φ₀) — на днях обучения (сессии < 01.01.Y) и на днях года Y (вне обучения обоих);
 • согласие z: |corr| между компонентами z Stage B (главные компоненты — по z этого энкодера на датах < 01.01.Y) и z Stage A, компоненты сопоставлены ОДНОЗНАЧНО по максимуму суммы |corr| (выравнивание, `linear_sum_assignment`), на датах года Y и на датах до Y.
Модуль не импортируется путём прогноза (`v7_4_cfg`, `v7_4_run` его не знают). Данные читаются только при вызове."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from models_v2.s14 import v7_1b_sta as STA, v7_1e_net as NET, v7_1e_pre as PRE, v7_4_cfg as C4

COLS = ["year", "seed", "pretext_A_train", "pretext_B_train", "pretext_A_year", "pretext_B_year", "corr1", "corr2", "corr3", "corr4", "corr_mean_year", "corr_mean_past"]
CHUNK = 400


def pretext_loss(base, pack: dict, rows: np.ndarray, eps: float) -> float:
    """MSE задачи-предлога (веса — валидные пары баров дня, как в `PRE._loss`) по дням `rows`, порциями."""
    y, ok = PRE.targets(pack, rows, eps)
    s = n = 0.0
    with PRE.deterministic(), torch.no_grad():
        for i in range(0, len(rows), CHUNK):
            r = rows[i:i + CHUNK]
            p = NET.forward_pred(base, NET.to_seq(pack["X"][r], pack["mask"][r]).double())[:, -NET.SLOTS:]
            w = torch.as_tensor(ok[i:i + CHUNK]).to(p.dtype)
            s, n = s + float((w * (p - torch.as_tensor(y[i:i + CHUNK]).to(p.dtype)) ** 2).sum()), n + float(w.sum())
    return s / max(n, 1.0)


def aligned_corr(zb: np.ndarray, za: np.ndarray) -> np.ndarray:
    """|corr| после выравнивания: (4,) — для каждой компоненты Stage A лучшая однозначно сопоставленная компонента Stage B. МУТАНТ «без выравнивания» (диагональ) даёт меньше при перестановке компонент."""
    ok = np.isfinite(zb).all(1) & np.isfinite(za).all(1)
    c = np.abs(np.corrcoef(za[ok].T, zb[ok].T)[:za.shape[1], za.shape[1]:])
    i, j = linear_sum_assignment(-c)
    return c[i, j]


def row(year: int, seed: int, pack: dict, index: pd.DatetimeIndex, art: dict, enc: dict) -> dict:
    """Одна строка таблицы: год и seed."""
    days, tr = pd.DatetimeIndex(pack["days"]), PRE.train_rows(pack, year)
    yr = np.flatnonzero((days >= pd.Timestamp(year, 1, 1)) & (days < pd.Timestamp(year + 1, 1, 1)) & pack["norm_ok"] & pack["mask"][:, -1].any(1))
    eps, A, B = float(art["meta"][0]), C4.base_from(art), C4.base_from(enc)
    zb, _ = C4.z_of_encoder(B, pack, index, year)
    za = np.full((len(index), 4), np.nan)
    pos = index.get_indexer(pd.DatetimeIndex(art["days"].astype("datetime64[ns]")))
    za[pos[pos >= 0]] = art["z"][pos >= 0]
    cy = aligned_corr(zb[index.year == year], za[index.year == year])
    cp = aligned_corr(zb[index < pd.Timestamp(year, 1, 1)], za[index < pd.Timestamp(year, 1, 1)])
    return dict(year=year, seed=seed, pretext_A_train=pretext_loss(A, pack, tr, eps), pretext_B_train=pretext_loss(B, pack, tr, eps), pretext_A_year=pretext_loss(A, pack, yr, eps),
                pretext_B_year=pretext_loss(B, pack, yr, eps), **{"corr%d" % (k + 1): float(v) for k, v in enumerate(cy)}, corr_mean_year=float(cy.mean()), corr_mean_past=float(cp.mean()))


def drift_table(pack: dict, index: pd.DatetimeIndex, enc_load, seeds=STA.SEEDS, years=C4.YEARS_B, art_load=STA.load, art_dir=None) -> pd.DataFrame:
    """Таблица (год × seed). enc_load(год, seed) -> словарь весов Stage B (`4/encB`); art_load — Stage A."""
    return pd.DataFrame([row(y, s, pack, index, art_load((art_dir or STA.directory()) / STA.name("tcn", y, s)), enc_load(y, s)) for y in years for s in seeds])[COLS]


def by_year(t: pd.DataFrame) -> dict:
    """Средние по seeds для CHECKPOINT `4.drift` (без p-значений): потери A и B, их разность, согласие z."""
    g = t.groupby("year").mean(numeric_only=True)
    return {str(y): dict(pretext_A_year=float(r.pretext_A_year), pretext_B_year=float(r.pretext_B_year), pretext_B_minus_A_train=float(r.pretext_B_train - r.pretext_A_train),
                         pretext_B_minus_A_year=float(r.pretext_B_year - r.pretext_A_year), corr_mean_year=float(r.corr_mean_year), corr_mean_past=float(r.corr_mean_past)) for y, r in g.iterrows()}
