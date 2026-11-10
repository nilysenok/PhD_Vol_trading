"""Y-B, признаки Y3/Y4 (план раздел 3 и 8, Ж-243). PIT: бары только дня d; медианы — по 20 прошлым сессиям < d, в которых слот присутствует; профиль Y4 — среднее доли r²_k/RV по 250 прошлым сессиям < d.
Склад M10 V7 `store_raw.npz` (паспорт `v7_0_seq.bar_table`): `lvol` = ln(volume) СЫРОЙ (np.log объёма бара до всякой нормировки; exp(lvol) — целые объёмы), `r` = 100·r, `r2` = r². Файлов не пишет.
φ_k: (1) k/52; (2) ln max(|r_k|, 1e-8) − медиана того же по слоту за 20 прошлых; (3) sign r_k; (4) ln(объём_k / медиана объёма слота за 20 прошлых). Нет прошлых наблюдений слота -> φ2 = φ4 = 0 (считаются в отчёте)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS, WIN, PROF_WIN, EPS = 53, 20, 250, 1e-8


def load_store() -> dict:
    """days, r, r2, lvol (D, 53), present; отсутствующие бары -> 0 (маска сохранена в present)."""
    from models_v2.s15 import de0_data as D
    b = np.load(D.OUT7 / "store_raw.npz", allow_pickle=True)
    ch = [str(c) for c in b["channels"]]
    pr = np.asarray(b["present"], bool)
    g = lambda c: np.where(pr, np.asarray(b["raw"][:, :, ch.index(c)], float), 0.0)
    return dict(days=pd.DatetimeIndex(b["days"]), r=g("r"), r2=g("r2"), lvol=g("lvol"), present=pr)


def feat_row(i: int) -> int:
    """Строка склада, из которой берутся бары признаков дня на позиции i: СТРОГО день d (мутация: i + 1 — бары дня t⁺)."""
    return i


def slot_roll(x: np.ndarray, present: np.ndarray, win: int, fn, lag: int = 1) -> np.ndarray:
    """(D, S) -> по слоту: fn от последних `win` наблюдений слота СТРОГО до сессии (lag = 1; мутация lag = 0 — с текущим днём). Нет наблюдений -> NaN."""
    out = np.full(x.shape, np.nan)
    for k in range(x.shape[1]):
        idx = np.flatnonzero(present[:, k])
        v = x[idx, k]
        for i in range(len(idx)):
            hi = i + 1 - lag
            lo = max(0, hi - win)
            if hi > lo:
                out[idx[i], k] = fn(v[lo:hi])
    return out


def rv_rows(st: dict) -> np.ndarray:
    return st["r2"].sum(1)


def phi_all(st: dict, lag: int = 1) -> tuple:
    """-> (Φ (D, 53, 4), число присутствующих баров без истории слота). Отсутствующие бары — нули."""
    pr, lr = st["present"], np.log(np.maximum(np.abs(st["r"]), EPS))
    vol = np.exp(st["lvol"])
    m_lr, m_v = slot_roll(lr, pr, WIN, np.median, lag), slot_roll(vol, pr, WIN, np.median, lag)
    no_hist = pr & np.isnan(m_lr)
    p2 = np.where(pr & ~np.isnan(m_lr), lr - m_lr, 0.0)
    p4 = np.where(pr & ~np.isnan(m_v), st["lvol"] - np.log(np.where(np.isnan(m_v) | (m_v <= 0), 1.0, m_v)), 0.0)
    p1 = np.where(pr, np.arange(SLOTS)[None, :] / (SLOTS - 1), 0.0)
    p3 = np.where(pr, np.sign(st["r"]), 0.0)
    return np.stack([p1, p2, p3, p4], axis=2), int(no_hist.sum())


def profile_all(st: dict, lag: int = 1) -> np.ndarray:
    """p̄ (D, 53): среднее доли r²_k/RV по ≤ 250 прошлым сессиям < d, где слот присутствует; нет истории -> 0."""
    pr, rv = st["present"], rv_rows(st)
    share = np.where(pr & (rv[:, None] > 0), st["r2"] / np.where(rv > 0, rv, 1.0)[:, None], 0.0)
    return np.nan_to_num(slot_roll(share, pr, PROF_WIN, np.mean, lag), nan=0.0)


def weights_fixed(pbar: np.ndarray, m: np.ndarray, n: np.ndarray) -> tuple:
    """Y4: w_k = n·p̄_k/Σ_{присутствующие} p̄; Σ p̄ = 0 -> w ≡ 1 (флаг). -> (w, число таких строк)."""
    e = pbar * m
    s = e.sum(1)
    bad = s <= 0
    w = np.where(bad[:, None], m.astype(float), n[:, None] * e / np.where(bad, 1.0, s)[:, None])
    return w, int(bad.sum())


def pair_data(st: dict, d, rv_target, x1_f, phi=None, pbar=None) -> tuple:
    """Данные пар по датам d: u = RV(t⁺)/X1, Φ, маска, r², n, ln RV_d, h4 (абляция Y4). -> (dict, диагностика)."""
    pos = st["days"].get_indexer(pd.DatetimeIndex(d))
    if (pos < 0).any():
        raise SystemExit("день d вне склада M10")
    rows = np.array([feat_row(int(p)) for p in pos])
    if phi is None:
        phi, nh = phi_all(st)
    else:
        phi, nh = phi
    pbar = profile_all(st) if pbar is None else pbar
    m, r2 = st["present"][rows], st["r2"][rows]
    n = m.sum(1).astype(float)
    rv = r2.sum(1)
    w4, nbad = weights_fixed(pbar[rows], m, n)
    lrv = np.log(rv)
    h4 = np.log((w4 * r2).sum(1)) - lrv
    dd = dict(u=np.asarray(rv_target, float) / np.asarray(x1_f, float), Phi=phi[rows], M=m, R2=r2, n=n, lrv=lrv, h4=h4)
    return dd, dict(n_nohist=nh, n_prof_bad=nbad, n_rv_zero=int((rv <= 0).sum()), mean_w4=float(np.mean((w4.sum(1) / n))))
