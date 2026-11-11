"""Проверки S-L-IMPL-1 (`l_walk_chk`): --check-equiv (L1.G4), --check-convex (L1.G5), --check-ctrl (L1.G6). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_model as CM
from models_v2.s18 import l_model as ML
from models_v2.s18 import l_walk as LW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as RC

ACT = RC.ACT["CK"]


def active_months(ctx: W.Ctx, k: int = 3, seed: int = 0) -> list:
    xd = pd.DatetimeIndex(ctx.x["d"])
    act = [(a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp(ACT[0])]
    return [act[int(i)] for i in np.random.default_rng(seed).choice(len(act), k, replace=False)]


def check_equiv(ctx: W.Ctx) -> list:
    bad, x, n = [], ctx.x, len(ctx.x)
    A = LW.arrays(ctx)
    Vm = CM.build_data(x, CF.load_feat()).view(np.arange(n))
    P = RL.pool_of("CK", ctx.D, x, W.month_bounds(ctx)[60][0])
    c, m = ML.stats_l("pi", A, P)
    for a in (-0.3, 0.1):
        v = torch.zeros(CM.NP, dtype=torch.float64)
        v[2], v[3] = a, a
        e1 = float(np.max(np.abs(ML.lnc(a, A["lnc8"]) - torch.log(Vm.c8 * CM.forward(v, Vm, "CK")["M"]).numpy())))
        V = ctx.view(P)
        l_r = float(RL.loss_r(v, V, "CK", 0.0, c, m, "R"))
        l_l = float(ML.loss_l(torch.tensor(a, dtype=torch.float64), V.rv, V.l_lnb, V.l_pi, V.l_bg, torch.log(V.c8), 0.0, c, m))
        print("a = %+.1f: max |ln cfg8 + a − ln(cfg8·𝓜_CK(γ = 0, θ_c0 = θ_p0 = a, θ₁ = 0))| = %.2e (2198 дней); |loss_l − RL.loss_r| = %.2e" % (a, e1, abs(l_r - l_l)))
        bad += ["эквивалентность a = %g: %.2e / %.2e" % (a, e1, abs(l_r - l_l))] * (e1 > 1e-12 or abs(l_r - l_l) > 1e-12)
    return bad


def check_convex(ctx: W.Ctx) -> list:
    bad, A = [], LW.arrays(ctx)
    for a0, b0 in active_months(ctx):
        P = RL.pool_of("CK", ctx.D, ctx.x, a0)
        c, m = ML.stats_l("pi", A, P)
        for lam in (1e-1, 1e-3):
            ah = [ML.fit_kind("pi", A, P, lam, c, m, s) for s in (-0.3, 0.0, 0.3)]
            sp = max(ah) - min(ah)
            print("месяц %s, λ = %g: â при стартах −0,3 / 0 / +0,3 = %s; разброс %.2e" % (pd.Timestamp(ctx.x["d"].iloc[a0]).date(), lam, ["%.10f" % z for z in ah], sp))
            bad += ["выпуклость %s λ = %g: разброс %.2e" % (a0, lam, sp)] * (sp > 1e-8)
    return bad


def _score0(A: dict, kind: str, T, c: float, m: float) -> float:
    T = np.asarray(T)
    t = lambda z: torch.as_tensor(np.asarray(z, float))
    a = torch.zeros((), dtype=torch.float64, requires_grad=True)
    f = ML.loss_l(a, t(A["rv"][T]), t(A["lnb"][T]), t(ML.train_pi(kind, A["pi"], T)), t(A["bg"][T]), t(A["lnc8"][T]), 0.0, c, m)
    f.backward()
    return abs(float(a.grad))


def check_ctrl(ctx: W.Ctx) -> list:
    bad, A, n = [], LW.arrays(ctx), len(ctx.x)
    for a0, b0 in active_months(ctx):
        P = RL.pool_of("CK", ctx.D, ctx.x, a0)
        for T in [P] + [t for t, _ in W.blocks(P)]:
            c, m = ML.stats_l("pibar", A, T)
            sc = _score0(A, "pibar", T, c, m)
            ah = [ML.fit_kind("pibar", A, T, lam, c, m) for lam in W.LAMS]
            bad += ["rl1a: балл в 0 = %.2e или â ≠ 0: %s" % (sc, ah)] * (sc > 1e-12 or any(z != 0.0 for z in ah))
            pib = np.array([A["pi"][(i - 252) % n] for i in T])
            xb = A["rv"][T] * np.exp(-((1.0 - pib) * A["lnb"][T] + pib * A["lnc8"][T] + A["bg"][T]))
            cb = RL.clip_stats(xb, np.arange(len(T)))
            okb = ML.stats_l("shift252", A, T) == cb
            bad += ["rl1b: c, m ≠ clip_stats(RV/F_train^(б)(0))"] * (not okb)
            Tn = np.asarray(T)[np.asarray(T) >= 252]
            pn = A["pi"][Tn - 252]
            xn = A["rv"][Tn] * np.exp(-((1.0 - pn) * A["lnb"][Tn] + pn * A["lnc8"][Tn] + A["bg"][Tn]))
            bad += ["rl1b2: c, m ≠ clip_stats по парам i ≥ 252 с π_{i−252}"] * (ML.stats_l("shift252nr", A, T) != RL.clip_stats(xn, np.arange(len(Tn))))
        print("месяц %s: rl1a балл в 0 по T = %.1e, â = 0 при всех λ; rl1b c, m = clip_stats(RV/F_train^(б)(0), T)" % (pd.Timestamp(ctx.x["d"].iloc[a0]).date(), sc))
        for kind in ("pibar", "shift252", "shift252nr"):
            lam, w, G = LW.choose_l(ctx, P, kind)
            err = 0.0
            for bi, (T, Vb) in enumerate(W.blocks(P)):
                c, m = ML.stats_l(kind, A, T)
                V, base = ctx.view(Vb), float(RL.ell(ctx.rv[Vb], ctx.lnf0[Vb]).mean())
                for l2 in W.LAMS:
                    a = ML.fit_kind(kind, A, T, l2, c, m)
                    lnf = RL.ln_f(torch.as_tensor(A["lnc8"][Vb] + a), V).numpy()
                    err = max(err, abs(float(RL.ell(ctx.rv[Vb], lnf).mean()) - base - G[l2][bi]))
            print("  %s: G месяца = QLIKE с настоящим π (пересчёт) max |Δ| = %.2e; λ* = %s" % (kind, err, lam))
            bad += ["%s: G месяца ≠ QLIKE с настоящим π (%.2e)" % (kind, err)] * (err > 1e-12)
    return bad
