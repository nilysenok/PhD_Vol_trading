"""Детекторы S-R-IMPL-2 на синтетике с известным ответом (для тестов и мутаций): правило выбора, блоки, закрытый пул, θ = 0 в сетке, валидация, включение, слои. Только расчёт, ничего не пишут.
Каждый детектор возвращает список замечаний; на честном коде — пустой, на мутации — непустой."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_w5_gap as W5
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_loss_syn as SY
from models_v2.s18 import r_walk as W

L1, L2, L3 = W.LAMS
BLOCK_CASES = [
    ({L1: [1, 1, 1], L2: [-1, -1, 1], L3: [1, 1, 1]}, L2), ({L1: [1, 1, 1], L2: [1, 1, 1], L3: [1, 1, 1]}, 0.0), ({L1: [-1, -1, -1], L2: [1, 1, 1], L3: [-1, -1, -1]}, L1),
    ({L1: [-1, 1, 1], L2: [-1, 1, 1], L3: [-1, 1, 1]}, 0.0), ({L1: [0, 0, 0], L2: [0, 0, 0], L3: [0, 0, 0]}, 0.0), ({L1: [-100, 1, 1], L2: [1, 1, 1], L3: [1, 1, 1]}, 0.0),
    ({L1: [-1, -1, 1], L2: [-1, -1, -1], L3: [-1, 1, 1]}, L2), ({L1: [-1, 1, -1], L2: [1, 1, 1], L3: [1, -1, -1]}, L1)]
SINGLE_CASES = [({L1: [0.5], L2: [-0.2], L3: [-0.1]}, L2), ({L1: [0.5], L2: [0.2], L3: [0.1]}, 0.0), ({L1: [-0.2], L2: [-0.2], L3: [0.1]}, L1), ({L1: [0.0], L2: [0.2], L3: [0.1]}, 0.0),
                ({L1: [0.5], L2: [-0.3], L3: [-0.3]}, L2)]


def rule_problems() -> list:
    bad = []
    for i, (G, want) in enumerate(BLOCK_CASES):
        lam, w = W.decide(G, "blocks")
        bad += ["правило блоков, случай %d: λ* %s ≠ %s" % (i, lam, want)] * (lam != want or w != {k: sum(g < 0 for g in G[k]) for k in G})
    for i, (G, want) in enumerate(SINGLE_CASES):
        bad += ["правило одной валидации, случай %d" % i] * (W.decide(G, "single")[0] != want)
    return bad


def block_problems() -> list:
    P, bad = np.arange(1000), []
    bl = W.blocks(P)
    bad += ["число блоков"] * (len(bl) != 3)
    for b, (T, V) in enumerate(bl):
        bad += ["блок %d: V ≠ 126 последовательных пар по порядку" % (b + 1)] * (not np.array_equal(V, P[len(P) - 126 * (3 - b): len(P) - 126 * (2 - b)]))
        bad += ["блок %d: T не раньше V" % (b + 1)] * (len(T) == 0 or T.max() >= V.min() or not np.array_equal(T, P[: V.min()]))
    return bad


def syn_ctx(n: int = 1500, seed: int = 0, ths=None, spike: int | None = None, sig: float = 0.5, jump: float = 0.01) -> W.Ctx:
    D, lv = SY.synth_data(n, seed, np.zeros(6) if ths is None else ths, sig=sig, jump=jump)
    if spike is not None:
        D.rv = D.rv.copy()
        D.rv[spike] *= 40.0
    d = pd.bdate_range("2018-01-01", periods=n + 1)
    x = pd.DataFrame(dict(d=d[:-1], t_plus=d[1:], C=D.c8, rv=D.rv))
    return W.Ctx("CK", D, lv, x, None)


def pool_problems() -> list:
    ctx, bad = syn_ctx(), []
    for a in (700, 701, 1200):
        P, d0 = RL.pool_of("CK", ctx.D, ctx.x, a), ctx.x["d"].iloc[a]
        want = np.flatnonzero(ctx.x["t_plus"].to_numpy() <= np.datetime64(d0))
        bad += ["пул CK a = %d: пары с t⁺ ≤ D не совпали" % a] * (not np.array_equal(P, want))
    tp = pd.bdate_range("2019-01-01", periods=300)
    stub = type("S", (), {"tp": tp.to_numpy()})()
    xs = pd.DataFrame(dict(d=tp - pd.offsets.BDay(1)))
    P = RL.pool_of("A", stub, xs, 250)
    bad += ["пул A: пары с t⁺ ≤ D не совпали"] * (not np.array_equal(P, np.flatnonzero(tp.to_numpy() <= np.datetime64(xs["d"].iloc[250]))))
    return bad


def zero_problems() -> list:
    """Без шума RV = F(0): θ = 0 обязан быть выбран, C_R = cfg8 побитно (θ = 0 в сетке)."""
    ctx = syn_ctx(1500, 4, sig=0.0, jump=0.0)
    r =W.fit_month(ctx, np.arange(1000), np.arange(1000, 1020), ctx.x["C"].to_numpy(float)[1000:1020], "R")
    return ["нулевой сигнал: λ* = %s ≠ 0, θ ≠ 0" % r["lam"]] * (r["lam"] != 0.0 or np.abs(r["theta"]).max() > 0) + ["C_R ≠ cfg8 при θ = 0"] * (not np.array_equal(r["C"], ctx.x["C"].to_numpy(float)[1000:1020]))


def val_problems() -> list:
    """G блока V₃ при λ = 0,1 равен независимому расчёту ОБЫЧНОЙ QLIKE на V₃ (в V₃ есть выброс x = 40)."""
    ctx = syn_ctx(1100, 3, ths=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3]), spike=950)
    P = np.arange(1000)
    G = W.choose(ctx, P, "R", "blocks")[2][L1][2]
    T, V = P[:874], P[874:]
    c, m = RL.clip_stats(ctx.x0, T)
    th = RL.fit_r(ctx.view(T), "CK", L1, c, m, "R")
    lnf = RL.ln_f(torch.as_tensor(RL.predict_lnc(th, ctx.view(V), "CK")), ctx.view(V)).numpy()
    q = lambda lf: float(np.mean(ctx.rv[V] * np.exp(-lf) + lf - np.log(ctx.rv[V]) - 1.0))
    want = q(lnf) - q(ctx.lnf0[V])
    return ["G блока V₃: %.3e ≠ независимому %.3e" % (G, want)] * (abs(G - want) > 1e-12)


def incl_problems() -> list:
    ctx, bad = syn_ctx(1500, 5), []
    c0 = ctx.x["C"].to_numpy(float)[700:710]
    for mode, rule in (("R", "blocks"), ("plain", "blocks"), ("R", "single")):
        act = [W.fit_month(ctx, np.arange(n), np.arange(700, 710), c0, mode, rule)["active"] for n in (629, 630)]
        bad += ["включение %s/%s: %s вместо [False, True] при |P| = 629 / 630" % (mode, rule, act)] * (act != [False, True])
    return bad


def layer_problems(ctx) -> list:
    """Слои X1 и W5 над C_R (не над cfg8): C_R ≠ cfg8 в одном месяце меняет X1_R и F; X1_R, F = независимый расчёт E1.run / W5.walk над C_R."""
    x, bad = ctx.x, []
    c8 = x["C"].to_numpy(float)
    parts = [(a, b, W.fit_month(ctx, np.arange(0), ctx.dp[a:b], c8[a:b], "R")) for a, b in W.month_bounds(ctx)]
    a, b, r = parts[60]
    r["C"] = r["C"] * 1.3
    df = W.assemble("ra2" if ctx.arch == "A" else "rck", ctx, parts)
    Cn = df["C"].to_numpy()
    e = E1.run(x["rv"].to_numpy(float), x["M"].to_numpy(float), Cn, x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy())
    w = W5.walk(x["d"], x["t_plus"], x["rv"], e["F"], ctx.gap, base_name="X1_A")
    bad += ["X1_R ≠ E1.run над C_R"] * (float(np.max(np.abs(df["X1_R"] / e["F"] - 1.0))) > 1e-12)
    bad += ["F ≠ W5.walk над X1_R"] * (float(np.max(np.abs(df["F"] / w["F"].to_numpy() - 1.0))) > 1e-12)
    bad += ["слой не реагирует на C_R"] * (not (df["X1_R"].to_numpy()[a:b] != x["F"].to_numpy(float)[a:b]).any())
    return bad
