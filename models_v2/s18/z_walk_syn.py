"""S-Z-IMPL-2, гейт Z2.G7 (решение дирижёра, журнал 12.11 01:20): нулевая синтетика полной процедуры S-RCKZ до регистрации. Только расчёт, ничего не пишет.
Реальные z̃ (zmaps, все 5 seeds), реальные пулы, блоки, сетка с θ = 0, правило «2 из 3», 630 закрытых пар; цель ln RV = ln F⁰ + шум без связи с z (логнормальный σ 0,5 и 1 % скачков ×30, E = 1, как `r_loss_syn`).
Критерий: QLIKE(R) − QLIKE(θ = 0) на датах после включения ≤ +0,10 % (seed шума 0; seeds 1, 2 — чувствительность, печать)."""
from __future__ import annotations

import torch  # noqa: F401  (первым, класс № 152)

import copy

import numpy as np
import pandas as pd

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_walk as ZW

TOL, SEEDS = 1e-3, (0, 1, 2)


def noise(n: int, seed: int, sig: float = 0.5, jump: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(sig * rng.normal(size=n) - sig ** 2 / 2) * np.where(rng.random(n) < jump, 30.0, 1.0) / (1.0 + 29.0 * jump)


def synth_ctx(ctx, seed: int):
    D = copy.copy(ctx.D)
    D.rv = np.exp(ctx.lnf0) * noise(len(ctx.lnf0), seed)
    return RW.Ctx("CK", D, ctx.lv, ctx.x, ctx.gap, ctx.dp)


def qlike_rel(cs, parts) -> tuple:
    """(среднее ℓ(F_R) / среднее ℓ(θ=0) − 1 на днях активных месяцев, [λ* по месяцам])."""
    e1, e0, lams = [], [], []
    for a, b, r in parts:
        if not r["active"]:
            continue
        idx, V = np.arange(a, b), cs.view(np.arange(a, b))
        lnf = RL.ln_f(torch.as_tensor(np.log(r["C"])), V).numpy()
        e1.append(RL.ell(cs.rv[idx], lnf))
        e0.append(RL.ell(cs.rv[idx], cs.lnf0[idx]))
        lams.append(r["lam"])
    return float(np.concatenate(e1).mean() / np.concatenate(e0).mean() - 1.0), lams


def synth_null() -> list:
    ctx, bad, (zm, _) = RC.make_ctx("CK"), [], ZW.load_z()
    for seed in SEEDS:
        cs = synth_ctx(ctx, seed)
        parts = ZW.walk(cs, zm, "CKZ", True, quiet=True)
        rel, lams = qlike_rel(cs, parts)
        nz = float(np.mean([l > 0 for l in lams]))
        vals, cnt = np.unique(np.array(lams), return_counts=True)
        print("шум seed %d%s: месяцев активных %d; QLIKE(R) − QLIKE(θ=0) = %+.4f %%; доля месяцев θ ≠ 0 = %.3f; λ* по значениям %s" % (seed, "" if seed == 0 else " (чувствительность)", len(lams), 100 * rel, nz, dict(zip(vals.tolist(), cnt.tolist()))))
        print("  λ* по месяцам:", [("%g" % l) for l in lams])
        if seed == 0 and rel > TOL:
            bad.append("Z2.G7: %+.4f %% > +0,10 %%" % (100 * rel))
    return bad
