"""V7-1a-E: предобучение энкодера (Stage A). Задача-предлог: ln(r²_{j+1}+ε) следующего бара по барам ≤ j; MSE на лог-метке, Adam,
ранняя остановка по последним 10 % дней обучающего прошлого. Энкодер года Y учится только на сессиях < 01.01.Y (`train_rows`).
ε = 0,01 · медиана положительных r² (`eps_pos`, бары обучающих дней; решение [V7] класса B, лист 1b). Метка — по последнему дню окна; предыдущие дни окна — контекст (все ≤ d).
Вызов на данных SBER — только из стадии 1b; здесь данные не читаются при импорте.
`bars` = pack: dict(X (D,W,53,11), mask (D,W,53), norm_ok (D,), days (D,) datetime64, r2 (D,53) сырой r², NaN где бара нет)."""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s14 import v7_0_seq as SQ
from models_v2.s14 import v7_1e_net as NET

EPS_REL, VAL_FRAC, MIN_ROWS = 1e-6, 0.10, 60
EPS_POS_REL = 0.01
R2 = SQ.CH.index("r2")


def make_pack(store: dict, W: int) -> dict:
    """pack из склада листа 0 (синтетика и тесты): тот же x_of, что строит X_w*.npz."""
    n = len(store["days"])
    X = np.zeros((n, W, SQ.SLOTS, SQ.K), np.float32)
    M, ok = np.zeros((n, W, SQ.SLOTS), bool), np.zeros(n, bool)
    for i in range(n):
        x, m, _, ok[i] = SQ.x_of(store, i, W)
        X[i], M[i] = x, m
    r2 = np.where(store["present"], store["raw"][:, :, R2], np.nan)
    return dict(X=X, mask=M, norm_ok=ok, days=np.asarray(store["days"], "datetime64[ns]"), r2=r2)


def pack_from_npz(W: int, out_dir) -> dict:
    """pack из сохранённых `X_w{W}.npz` и `store_raw.npz` листа 0. Читает файлы только при вызове (стадия 1b+)."""
    out_dir = Path(out_dir)
    a, b = np.load(out_dir / ("X_w%d.npz" % W)), np.load(out_dir / "store_raw.npz")
    r2 = np.where(b["present"], b["raw"][:, :, R2], np.nan)
    return dict(X=a["X"], mask=a["mask"], norm_ok=a["norm_ok"], days=a["days"], r2=r2)


def train_rows(pack: dict, year: int) -> np.ndarray:
    """Строки дней обучения энкодера года Y: сессии строго до 01.01.Y, нормировка зрелая, у дня есть бары."""
    before = pd.DatetimeIndex(pack["days"]) < pd.Timestamp(year, 1, 1)
    return np.flatnonzero(before & pack["norm_ok"] & pack["mask"][:, -1].any(1))


def eps_pos(pack: dict, rows: np.ndarray) -> float:
    """ε метки (решение [V7] класса B, до листа 1b): 0,01·медиана ПОЛОЖИТЕЛЬНЫХ r² баров обучающих дней `rows` (только прошлое: rows = `train_rows`). Нет положительных — EPS_REL."""
    v = pack["r2"][rows]
    v = v[np.isfinite(v) & (v > 0)]
    return EPS_POS_REL * float(np.median(v)) if len(v) else EPS_REL


def targets(pack: dict, rows: np.ndarray, eps: float) -> tuple:
    """-> y (N, 53) = ln(r²_{s+1}+ε), valid (N, 53): бар s и бар s+1 того же дня есть."""
    r = pack["r2"][rows]
    y, ok = np.zeros(r.shape), np.zeros(r.shape, bool)
    ok[:, :-1] = np.isfinite(r[:, :-1]) & np.isfinite(r[:, 1:])
    y[:, :-1] = np.log(np.where(ok[:, :-1], r[:, 1:], 0.0) + eps)
    return y.astype(np.float32), ok


@contextlib.contextmanager
def deterministic():
    """1 поток, детерминированные алгоритмы; прежнее состояние восстанавливается."""
    thr, det = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det)
        torch.set_num_threads(thr)


def _loss(model, x, y, ok):
    p = NET.forward_pred(model, x)[:, -NET.SLOTS:]
    w = ok.to(p.dtype)
    return (w * (p - y) ** 2).sum() / w.sum().clamp(min=1.0)


def _batch(pack, rows, y, ok, idx):
    r = rows[idx]
    return NET.to_seq(pack["X"][r], pack["mask"][r]), torch.as_tensor(y[idx]), torch.as_tensor(ok[idx])


def train_encoder(kind: str, window: int, year: int, seed: int, bars: dict, max_epochs: int = 30, patience: int = 4,
                  lr: float = 1e-2, batch: int = 32, min_rows: int = MIN_ROWS):
    """Энкодер года `year` (kind tcn|cnn, окно 1|5 дней, seed 0–4) по барам сессий < 01.01.year. -> модель (eval) с `.meta`."""
    if bars["X"].shape[1] != window:
        raise ValueError("окно pack не равно window")
    rows = train_rows(bars, year)
    if len(rows) < min_rows:
        raise ValueError("мало дней для предобучения года %d: %d" % (year, len(rows)))
    nv = max(1, int(round(VAL_FRAC * len(rows))))
    eps = eps_pos(bars, rows)
    y, ok = targets(bars, rows, eps)
    n_tr = len(rows) - nv
    with deterministic():
        torch.manual_seed(seed)
        model, g = NET.build(kind), torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        xv, yv, okv = _batch(bars, rows, y, ok, np.arange(n_tr, len(rows)))
        best, best_state, bad, ep = np.inf, copy.deepcopy(model.state_dict()), 0, 0
        for ep in range(1, max_epochs + 1):
            model.train()
            for b in torch.randperm(n_tr, generator=g).split(batch):
                opt.zero_grad()
                _loss(model, *_batch(bars, rows, y, ok, b.numpy())).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(_loss(model, xv, yv, okv))
            if v < best - 1e-9:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
    model.meta = dict(kind=kind, window=window, year=year, seed=seed, eps=eps, n_train=n_tr, n_val=nv, epochs=ep, best_val=best,
                      train_days=np.asarray(bars["days"])[rows], n_params=NET.n_params(model))
    return model
