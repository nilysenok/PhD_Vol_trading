"""Этап Z, признаки затвора (план Z раздел 3; PIT: только бары дня d склада M10 V7 `store_raw.npz` и RV ПРОШЛЫХ дней). Файлов не пишет.
x_d = ln max_k r²_{d,k} − ln(RV_d/n_d) — по присутствующим барам дня d (n_d — их число; RV_d = Σ r² присутствующих баров, то же определение, что цель);
v_d = ln RV_d − ln mean(RV_{d−1..d−20}) — по 20 торговым дням склада перед d (день d и d + 1 в окно не входят);
z_d = (e_d, x_d, s_d, v_d); z̃ — стандартизация средним и σ (ddof = 0) по парам ПУЛА переоценки (t⁺ ≤ D); ZCp — z_{d−252} циклическим сдвигом по ряду общих дат (np.roll).
Дни с RV_d ≤ 0 или n_d = 0: x_d = v_d = 0 (счётчик в отчёте)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_ya_feat as FE

WIN, SHIFT = 20, 252
load_store = FE.load_store


def feat_row(i: int) -> int:
    """Строка склада бара признаков дня на позиции i: СТРОГО день d (мутация: i + 1 — бары дня t⁺)."""
    return i


def hist_rows(p: int, win: int = WIN) -> np.ndarray:
    """Строки склада RV_{d−1}…RV_{d−20} (мутация: окно доходит до дня d + 1)."""
    return np.arange(p - win, p)


def store_stats(st: dict) -> tuple:
    """-> (RV, n, ln max r²) по строкам склада (присутствующие бары)."""
    r2 = np.where(st["present"], st["r2"], 0.0)
    return r2.sum(1), st["present"].sum(1).astype(float), r2.max(1)


def x_v(st: dict, d) -> tuple:
    """x_d, v_d, число дней с вырожденным RV для дат d (все — в складе, ≥ 20 дней истории)."""
    pos = st["days"].get_indexer(pd.DatetimeIndex(d))
    if (pos < 0).any() or (pos < WIN).any():
        raise SystemExit("день d вне склада M10 или меньше %d дней истории" % WIN)
    rv, n, mx = store_stats(st)
    x, v, bad = np.zeros(len(pos)), np.zeros(len(pos)), 0
    for i, p in enumerate(pos):
        r, h = feat_row(int(p)), hist_rows(int(p))
        mh = rv[h].mean()
        if rv[r] <= 0 or n[r] <= 0 or mx[r] <= 0 or mh <= 0:
            bad += 1
            continue
        x[i], v[i] = np.log(mx[r]) - np.log(rv[r] / n[r]), np.log(rv[r]) - np.log(mh)
    return x, v, bad


def raw_z(e, x, s, v) -> np.ndarray:
    """(n, 4): e_d, x_d, s_d, v_d."""
    return np.column_stack([np.asarray(e, float), x, np.asarray(s, float), v])


def shift_z(Z: np.ndarray, k: int = SHIFT) -> np.ndarray:
    """ZCp: z_{d−k} циклическим сдвигом по ряду общих дат (мутация: без сдвига)."""
    return np.roll(Z, k, axis=0)


def std_stats(Z: np.ndarray, pool: np.ndarray) -> tuple:
    """Среднее и σ по парам пула (мутация: по всей выборке)."""
    return Z[pool].mean(0), Z[pool].std(0)


def standardize(Z: np.ndarray, pool: np.ndarray) -> np.ndarray:
    mu, sd = std_stats(Z, pool)
    return (Z - mu) / np.where(sd > 1e-12, sd, 1.0)
