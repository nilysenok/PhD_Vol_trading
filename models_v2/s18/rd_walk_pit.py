"""Проверки S-RCD-IMPL-2 (`rd_walk`): --check-pit (W.G3), --check-level (W.G5). Только чтение; --check-level читает выходы `RD/*.parquet`."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_model as RM
from models_v2.s18 import rd_model_chk as MC
from models_v2.s18 import rd_walk as RD
from models_v2.s18.rd_walk_chk import ctx_of

KEYS = ("lam", "w", "G", "c", "m", "theta", "C", "val", "spread")


def pool_of(var: str, ctx, a: int):
    if RD.VARS[var][1] == "R":
        return RL.pool_of("CK", ctx.D, ctx.x, a)
    xd = pd.DatetimeIndex(ctx.x["d"])
    return np.asarray(YA.closed_pool(pd.DatetimeIndex(ctx.x["t_plus"]), xd, xd[a]))


def spoil_rv(D, mask):
    Dc = copy.copy(D)
    Dc.rv = np.where(mask, np.asarray(D.rv, float) * 7.0, D.rv)
    return Dc


def with_data(ctx, D, lv):
    return W.Ctx("CK", D, lv, ctx.x, ctx.gap, ctx.dp)


def same(a: dict, b: dict) -> bool:
    return all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float), equal_nan=True) for k in KEYS)


def block_fit(var: str, ctx, P):
    """θ̂ обучающей подгонки без последнего блока валидации: rcd — θ̂_{1e-2, b = 3} на T_3 (c, m по T_3); cdf, ckf — подгонка на P без последних 126, λ = 1e-2."""
    form, proto = RD.VARS[var]
    if proto == "R":
        T = W.blocks(P)[-1][0]
        c, m = W.stats_of(ctx.x0, T, "R")
        return RM.fit(ctx.view(T), form, 1e-2, c, m, "R")[0]
    return RM.fit(ctx.view(P[:-RD.CW.VAL]), form, 1e-2, float("inf"), 1.0, "plain")[0]


def check_pit(n_months: int = 3, seed: int = 0) -> list:
    bad, rng, ctx = [], np.random.default_rng(seed), ctx_of()
    xd, mb = pd.DatetimeIndex(ctx.x["d"]), W.month_bounds(ctx)
    act = [(a, b) for a, b in mb if xd[a] >= pd.Timestamp("2020-08-03") and b < len(xd)]
    n = len(ctx.lv.pi)
    for k in rng.choice(len(act) - 1, n_months, replace=False):
        a, b = act[int(k)]
        for var in RD.VARS:
            P = pool_of(var, ctx, a)
            out = np.ones(n, bool)
            out[P] = False
            late = np.arange(n) >= b
            Dc1 = spoil_rv(ctx.D, out)                                  # RV пар t⁺ > D
            Dc2, lc2 = MC.corrupt(ctx.D, ctx.lv, late)                  # строки feat, π, β дней после конца месяца
            Dc3 = spoil_rv(Dc2, out)
            base = RD.month(var, ctx, a, b)
            r = [RD.month(var, with_data(ctx, d_, l_), a, b) for d_, l_ in ((Dc1, ctx.lv), (Dc2, lc2), (Dc3, lc2))]
            blk = spoil_rv(ctx.D, np.isin(np.arange(n), P[-RD.CW.VAL:]))
            cb = with_data(ctx, blk, ctx.lv)
            blk_ok = np.array_equal(block_fit(var, ctx, P), block_fit(var, cb, P))
            ok = all(same(base, x) for x in r) and blk_ok
            print("PIT %s месяц D = %s: λ, w, θ̂, C при порче RV пар t⁺ > D / строк после месяца / обоих: %s; порча RV блока валидации не меняет θ̂ обучающей подгонки: %s; λ = %s" % (var, xd[a].date(), [same(base, x) for x in r], blk_ok, base["lam"]))
            bad += ["порча будущего изменила %s %s" % (var, xd[a].date())] * (not ok)
    return bad


def proj_grad(th, V, form, lam, c, m, mode) -> float:
    v = torch.as_tensor(th).clone().requires_grad_(True)
    RM.loss(v, V, form, lam, c, m, mode).backward()
    g, (lo, hi) = v.grad.numpy().copy(), np.array(CM.bounds(form)).T
    g[((th <= lo + 1e-12) & (g > 0)) | ((th >= hi - 1e-12) & (g < 0))] = 0.0
    return float(np.max(np.abs(g)))


def check_level() -> list:
    bad, ctx, cc = [], ctx_of(), ctx_of("C")
    xd = pd.DatetimeIndex(ctx.x["d"])
    for var, (form, proto) in RD.VARS.items():
        p = RD.RD_DIR / RD.OUTS[var]
        if not p.exists():
            bad.append("нет %s" % p.name)
            continue
        df = pd.read_parquet(p)
        for s in ("2021-03-01", "2023-03-01", "2026-11-01"):
            a = int(np.flatnonzero(xd >= pd.Timestamp(s))[0])
            while proto == "R" and not df["lam"].iloc[a] > 0:    # θ = 0 в сетке: стационарности нет — следующий месяц с λ* > 0
                a = min(b for _, b in W.month_bounds(ctx) if b > a)
            th, lam = df[list(CM.NAMES[form])].iloc[a].to_numpy(float), float(df["lam"].iloc[a])
            P = pool_of(var, ctx, a)
            c, m = (float(df["c"].iloc[a]), float(df["m"].iloc[a])) if proto == "R" else (float("inf"), 1.0)
            mode = "R" if proto == "R" else "plain"
            gf = proj_grad(th, ctx.view(P), form, lam, c, m, mode)
            gc = proj_grad(th, cc.view_train(P), form, lam, float("inf"), 1.0, "plain")
            print("  %s %s: |P| = %d, λ = %g; проекц. градиент заявленной потери (уровень F) %.2e; потери уровня C %.2e" % (var, xd[a].date(), len(P), lam, gf, gc))
            bad += ["%s %s: градиент заявленной потери %.2e" % (var, xd[a].date(), gf)] * (gf > 1e-4) + ["%s %s: уровень C %.2e" % (var, xd[a].date(), gc)] * (gc < 1e-3)
    return bad
