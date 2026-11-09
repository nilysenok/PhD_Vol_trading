"""V7-1a-M, обучение по QLIKE (план п. 4). Adam, полный батч. Строки обучения — ЗАКРЫТЫЕ пары (отбирает walk-forward), внутренняя валидация — последние 126 из них.
Для каждой точки решётки λ_g ∈ {0,1; 1} × λ_LB ∈ {0; 0,1} и каждого seed 0–4 голова учится на обучающей части с ранней остановкой по валидации (лучшая эпоха); точка решётки выбирается по QLIKE
ансамбля (среднее ln F по seeds) на валидации; затем 5 seeds переобучаются на ВСЕХ закрытых парах с числом эпох = лучшая эпоха своего seed. Прогноз = среднее ln F по seeds.
Данные SBER модуль не читает; вызывается walk-forward."""
from __future__ import annotations

import copy

import numpy as np
import torch

from models_v2.s14 import v7_1m_forms as FM, v7_1m_loss as LS

LR, MAX_EP, PATIENCE = 0.02, 300, 25
SEEDS = (0, 1, 2, 3, 4)
GRID = tuple((lg, lb) for lg in (0.1, 1.0) for lb in (0.0, 0.1))      # λ_g × λ_LB, 4 точки (в N_wide)
KEYS = ("Zg", "z", "ell", "H", "rv")


def batch(Zg, z, ell, H, rv) -> dict:
    return {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64) for k, v in zip(KEYS, (Zg, z, ell, H, rv))}


def take(b: dict, idx) -> dict:
    ix = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return {k: v[ix] for k, v in b.items()}


def cat(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


def make(spec: dict, seed: int):
    """spec: kind 'moe' | 'anchor' | 'flat'; router 'A'|'B'|'C'; K; z_only (плоский на z); encoder (фабрика nn.Module, совместное обучение Stage B)."""
    k = spec["kind"]
    if k == "flat":
        head = FM.Flat(FM.DZ if spec.get("z_only") else FM.DZ_GATE + FM.DZ, seed)
    else:
        head = FM.ResMoE(spec["router"], spec.get("K", 2), anchor=(k == "anchor"), seed=seed)
    if spec.get("encoder") is None:
        return head
    enc = spec["encoder"]().double()
    return LS.Joint(head, enc, spec.get("lam_e", LS.REG["lam_e"]))


def lnF(model, b: dict) -> torch.Tensor:
    with torch.no_grad(), FM.deterministic():                # один поток: с несколькими потоками softmax падает после импорта lightgbm (общий OpenMP macOS)
        return torch.log(b["H"]) + model(b["Zg"], b["z"], b["ell"])[0]


def train_one(spec, seed, lam_g, lam_lb, tr: dict, va: dict | None = None, epochs: int | None = None):
    """-> (модель, лучшая эпоха, QLIKE на va или nan). С va — ранняя остановка (терпение PATIENCE); без va — ровно `epochs` шагов."""
    model = make(spec, seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_ep, best_state, bad = float("inf"), 0, None, 0
    for ep in range(1, (epochs if va is None else MAX_EP) + 1):
        opt.zero_grad()
        LS.loss(model, tr, lam_g, lam_lb).backward()
        opt.step()
        if va is None:
            continue
        with torch.no_grad():
            q = float(LS.qlike_np(va["rv"], va["H"] * torch.exp(model(va["Zg"], va["z"], va["ell"])[0])))
        if q < best:
            best, best_ep, best_state, bad = q, ep, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if va is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best if va is not None else float("nan")


def ensemble_lnF(models, b: dict) -> torch.Tensor:
    with FM.deterministic():
        return torch.stack([lnF(m, b) for m in models]).mean(dim=0)


def val_qlike(models, va: dict) -> float:
    return float(LS.qlike_np(va["rv"], torch.exp(ensemble_lnF(models, va))))


def fit_config(spec: dict, tr: dict, va: dict, seeds=SEEDS, grid=GRID) -> dict:
    """-> {models: переобученные на tr ∪ va, lam: (λ_g, λ_LB), val: таблица решётки, epochs}. Решётка и ранняя остановка — только по va (последние 126 закрытых пар)."""
    rows = []
    with FM.deterministic():
        for lg, lb in grid:
            res = [train_one(spec, s, lg, lb, tr, va) for s in seeds]
            rows.append(dict(lam_g=lg, lam_lb=lb, q=val_qlike([r[0] for r in res], va), epochs=[r[1] for r in res]))
        best = min(rows, key=lambda r: r["q"])
        full = cat(tr, va)
        models = [train_one(spec, s, best["lam_g"], best["lam_lb"], full, None, max(1, e))[0] for s, e in zip(seeds, best["epochs"])]
    return dict(models=models, lam=(best["lam_g"], best["lam_lb"]), val=rows, epochs=best["epochs"])
