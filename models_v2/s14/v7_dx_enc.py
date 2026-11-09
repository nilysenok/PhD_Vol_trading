"""V7-DX, G37 (а): выразительность энкодера. Сохранённые энкодеры Stage A (`stageA/<kind>_w5_<Y>_s<seed>.npz`, веса лучшей эпохи) НЕ переобучаются.
1) Задача-предлог на отложенных 10 % прошлого года (последние `nv` дней `train_rows`, как при предобучении): MSE на ln(r²_{j+1}+ε) энкодера против трёх наивных прогнозов, считаемых по прошлому (бары ≤ j; дни строго до):
   mean — среднее ln(r²+ε) баров дня до j включительно; ewma — EWMA баров дня (α = 0,2, объявлено); profile — среднее метки слота j по прошлым дням (расширяющееся); const — среднее метки по обучающей части (контроль).
2) Ранг 16-мерного представления на днях < Y: доля дисперсии 4 главных компонент (стандартизованные оси, корреляционная матрица) и эффективный ранг (Σλ)²/Σλ².
3) Стабильность z между seeds одного года на днях года Y: средняя |corr| после выравнивания (венгерский алгоритм по |corr|). Данные читаются при вызове функций."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from models_v2.s14 import v7_0_io as IO, v7_1b_sta as STA, v7_1e_net as NET, v7_1e_pre as PRE, v7_1e_rep as REP

ALPHA_EWMA = 0.2
SLOTS = NET.SLOTS if hasattr(NET, "SLOTS") else 53


def naive_forecasts(r2: np.ndarray, y: np.ndarray, ok: np.ndarray, eps: float, n_tr: int) -> dict:
    """r2 (N, 53) сырые r² (NaN — бара нет), y/ok — метка и годность (`PRE.targets`), строки отложенной части — [n_tr:]. -> {имя: (N - n_tr, 53)} прогнозы метки на слоте j только по прошлому."""
    fin = np.isfinite(r2)
    L = np.where(fin, np.log(np.where(fin, r2, 1.0) + eps), np.nan)
    N, S = L.shape
    Lh, fh = L[n_tr:], fin[n_tr:]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.cumsum(np.where(fh, Lh, 0.0), axis=1) / np.cumsum(fh, axis=1)      # среднее баров дня ≤ j (NaN, пока баров нет)
    ew, a = np.full(Lh.shape, np.nan), np.full(Lh.shape[0], np.nan)
    for s in range(S):
        upd = fh[:, s]
        a = np.where(upd, np.where(np.isfinite(a), (1 - ALPHA_EWMA) * a + ALPHA_EWMA * np.where(upd, Lh[:, s], 0.0), Lh[:, s]), a)
        ew[:, s] = a
    cs, cn = np.cumsum(np.where(ok, y, 0.0), axis=0), np.cumsum(ok, axis=0)
    prof = np.full((N, S), np.nan)
    prof[1:] = cs[:-1] / np.maximum(cn[:-1], 1)                      # дни строго до i
    const = float(y[:n_tr][ok[:n_tr]].mean())
    return dict(mean=mean, ewma=ew, profile=prof[n_tr:], const=np.full((N - n_tr, S), const))


def mse(p: np.ndarray, y: np.ndarray, ok: np.ndarray) -> float:
    m = ok & np.isfinite(p)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else float("nan")


def load_model(kind: str, year: int, seed: int):
    a = STA.load(STA.directory() / STA.name(kind, year, seed))
    m = NET.build(kind)
    m.load_state_dict({k[3:]: torch.as_tensor(v) for k, v in a.items() if k.startswith("w__")})
    return m.eval(), a


def holdout_row(kind: str, year: int, seed: int, pack: dict) -> dict:
    m, a = load_model(kind, year, seed)
    rows = PRE.train_rows(pack, year)
    nv = max(1, int(round(PRE.VAL_FRAC * len(rows))))
    n_tr, eps = len(rows) - nv, float(a["meta"][0])
    y, ok = PRE.targets(pack, rows, eps)
    with PRE.deterministic(), torch.no_grad():
        p = NET.forward_pred(m, NET.to_seq(pack["X"][rows[n_tr:]], pack["mask"][rows[n_tr:]]))[:, -NET.SLOTS:].numpy().astype(float)
    nf = naive_forecasts(pack["r2"][rows], y, ok, eps, n_tr)
    yh, oh = y[n_tr:].astype(float), ok[n_tr:]
    out = dict(kind=kind, year=year, seed=seed, n_holdout_days=nv, mse_enc=mse(p, yh, oh), best_val_saved=float(a["meta"][4]))
    out.update({"mse_" + k: mse(v, yh, oh) for k, v in nf.items()})
    out["enc_beats_all_naive"] = bool(out["mse_enc"] < min(out["mse_mean"], out["mse_ewma"], out["mse_profile"]))
    return out


def rank_row(kind: str, year: int, seed: int, pack: dict) -> dict:
    m, a = load_model(kind, year, seed)
    rows = PRE.train_rows(pack, year)
    raw = REP.encode(m, pack["X"][rows], pack["mask"][rows])
    sd = raw.std(0)
    live = sd > 1e-12
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(raw[:, live], rowvar=False)))[::-1] if live.sum() > 1 else np.array([1.0])
    ev = np.clip(ev, 0, None)
    return dict(kind=kind, year=year, seed=seed, n_dead_axes=int((~live).sum()), share_top4=float(ev[:4].sum() / ev.sum()), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()))


def stability(zs: list) -> dict:
    """zs: список (n, 4) z разных seeds на одних датах. -> средняя и наименьшая |corr| после выравнивания перестановкой; средняя |corr| по диагонали без выравнивания."""
    al, raw = [], []
    for x, y in itertools.combinations(zs, 2):
        c = np.abs(np.corrcoef(x, y, rowvar=False)[:4, 4:])
        r, k = linear_sum_assignment(-c)
        al.append(c[r, k].mean())
        raw.append(np.diag(c).mean())
    return dict(aligned_mean=float(np.mean(al)), aligned_min=float(np.min(al)), unaligned_mean=float(np.mean(raw)))


def stability_row(kind: str, year: int) -> dict:
    zs = []
    for s in STA.SEEDS:
        a = STA.load(STA.directory() / STA.name(kind, year, s))
        d = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
        zs.append(a["z"][np.asarray((d >= pd.Timestamp(year, 1, 1)) & (d < pd.Timestamp(year + 1, 1, 1)))])
    return dict(kind=kind, year=year, n_days=len(zs[0]), **stability(zs))


def tables(pack: dict) -> dict:
    """-> DX_ENC (по kind, году, seed: потери и ранг) и DX_ENC_STAB (по kind, году)."""
    hold = [{**holdout_row(k, y, s, pack), **{c: v for c, v in rank_row(k, y, s, pack).items() if c not in ("kind", "year", "seed")}} for k in STA.KINDS for y in STA.YEARS for s in STA.SEEDS]
    stab = [stability_row(k, y) for k in STA.KINDS for y in STA.YEARS]
    return dict(ENC=pd.DataFrame(hold), ENC_STAB=pd.DataFrame(stab))


def load_pack() -> dict:
    return PRE.pack_from_npz(STA.WINDOW, IO.OUT)
