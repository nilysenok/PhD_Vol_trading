"""Детекторы S-D-IMPL-1 на синтетике с известным ответом (для тестов и мутаций): S-RCKm (m = 1, c = Q₀,₉₉ RV/W5-G) и S-RCKf (c, m, потеря обучения на уровне C; валидация — QLIKE уровня F).
Расчёт ожидаемого ничего не берёт из `Ctx` кроме входных данных: уровень F, x0, c, m, θ̂ и G блока пересчитываются заново `numpy` и `RL.fit_r`. Честный код — пустой список замечаний, мутация — непустой. Ничего не пишут."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import d_walk as D
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_loss_syn as SY
from models_v2.s18 import r_walk as W

L1 = W.LAMS[0]
THS = np.array([0.0, 0.5, -0.05, -0.05, -0.6, 0.6])


def lv_c(n: int) -> RL.Lv:
    return RL.Lv(np.ones(n), np.zeros(n), np.zeros(n), np.ones(n))


def syn_ctx(level: str = "F", n: int = 1100, seed: int = 0, ths=THS, train=None) -> W.Ctx:
    """Синтетика, где уровень F ≠ уровню C: π случайный, ln B ≠ ln cfg8, β·g ≠ 0 в части дней; RV = F(θ*)·Z (Z — шум SY.synth_data)."""
    Dd, lv0 = SY.synth_data(n, seed, np.asarray(ths, float))
    rng = np.random.default_rng(seed + 99)
    lv = RL.Lv(rng.uniform(0.25, 0.75, n), np.log(Dd.c8) + 0.15 * rng.normal(size=n), 0.03 * rng.normal(size=n) * (rng.random(n) < 0.2), np.ones(n))
    th = torch.as_tensor(np.asarray(ths, float))
    f = lambda L: torch.exp(RL.ln_f(RL.lnc_of(th, RL.view(Dd, L, np.arange(n)), "CK"), RL.view(Dd, L, np.arange(n)))).numpy()
    Dd.rv = f(lv) * (Dd.rv / f(lv0))
    d = pd.bdate_range("2018-01-01", periods=n + 1)
    x = pd.DataFrame(dict(d=d[:-1], t_plus=d[1:], C=Dd.c8, rv=Dd.rv))
    use_c = (level == "C") if train is None else train
    lt = lv_c(n) if use_c else None
    return W.Ctx("CK", Dd, lv, x, None, lv_train=lt)


def stats_expected(var: str, ctx: W.Ctx, T) -> tuple:
    lv, c8 = ctx.lv, ctx.x["C"].to_numpy(float)
    lnf0 = (1 - lv.pi) * lv.lnb + lv.pi * np.log(c8) + lv.bg
    x0 = ctx.rv / c8 if var == "rckf" else ctx.rv * np.exp(-lnf0)
    c = float(np.quantile(x0[T], 0.99))
    return c, (1.0 if var == "rckm" else float(np.mean(np.minimum(x0[T], c))))


def problems(var: str, ctx: W.Ctx) -> list:
    n, P = len(ctx.rv), np.arange(1000)
    lv, c8, bad = ctx.lv, ctx.x["C"].to_numpy(float), []
    lt = lv_c(n) if var == "rckf" else lv
    r = W.fit_month(ctx, P, ctx.dp[1000:1020], c8[1000:1020], D.VARS_D[var][1])
    c, m = stats_expected(var, ctx, P)
    bad += ["%s: c, m месяца = %s, %s ≠ ожидаемым %s, %s" % (var, r["c"], r["m"], c, m)] * (not (r["active"] and np.isclose(r["c"], c, rtol=1e-12, atol=0) and np.isclose(r["m"], m, rtol=1e-12, atol=0)))
    T, V = P[:874], P[874:]
    cT, mT = stats_expected(var, ctx, T)
    th = RL.fit_r(RL.view(ctx.D, lt, T), "CK", L1, cT, mT, "R")
    lnf = RL.ln_f(torch.as_tensor(RL.predict_lnc(th, RL.view(ctx.D, lv, V), "CK")), RL.view(ctx.D, lv, V)).numpy()
    lnf0 = (1 - lv.pi) * lv.lnb + lv.pi * np.log(c8) + lv.bg
    want = float(RL.ell(ctx.rv[V], lnf).mean() - RL.ell(ctx.rv[V], lnf0[V]).mean())
    bad += ["%s: G блока V₃ = %.3e ≠ независимому QLIKE уровня F %.3e" % (var, r["G"][0, 2], want)] * bool(abs(r["G"][0, 2] - want) > 1e-8)
    if r["lam"] > 0:
        thf = np.mean(RL.fit_r(RL.view(ctx.D, lt, P), "CK", r["lam"], c, m, "R"), axis=0)
        bad += ["%s: θ̂ месяца ≠ подгонке на независимом уровне обучения" % var] * (not np.allclose(thf, r["theta"], rtol=0, atol=1e-6))
    sc = level_score(RL.view(ctx.D, lt, P), c, m)
    bad += ["rckf: балл уровня в θ = 0 на уровне C = %.2e ≠ 0" % sc] * bool(var == "rckf" and abs(sc) > 1e-12)
    return bad


def saved_month_problems(make_ctx=None) -> list:
    """Умолчания = прежний S-RCK / S-RCKb: месяц 2022-04-29, C побитно = сохранённому `R/rck.parquet`, `R/rckb.parquet` (боевые данные)."""
    ctx, bad = (make_ctx or W.make_ctx)("CK"), []
    xd = pd.DatetimeIndex(ctx.x["d"])
    a, b = [(a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp("2022-04-01")][0]
    for var, mode in (("rck", "R"), ("rckb", "plain")):
        r = W.fit_month(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], ctx.x["C"].to_numpy(float)[a:b], mode)
        bad += ["%s: месяц %s ≠ сохранённому parquet" % (var, xd[a].date())] * (not np.array_equal(r["C"], pd.read_parquet(W.R_DIR / (var + ".parquet"))["C"].to_numpy()[a:b]))
    return bad


def assemble_problems() -> list:
    """`assemble` для id вне `W.VARS` (rckf): архитектура — из `ctx.arch`; при θ = 0 C = cfg8 побитно (боевые данные, пустой пул)."""
    ctx = W.make_ctx("CK", "C")
    c8 = ctx.x["C"].to_numpy(float)
    parts = [(a, b, W.fit_month(ctx, np.arange(0), ctx.dp[a:b], c8[a:b], "R")) for a, b in W.month_bounds(ctx)]
    try:
        df = W.assemble("rckf", ctx, parts)
    except KeyError as e:
        return ["assemble: KeyError %s (архитектура из VARS[var])" % e]
    return [] if len(df) == 2198 and np.array_equal(df["C"].to_numpy(), c8) and not df["active"].any() else ["assemble: состав ≠ ожидаемому"]


def level_score(V, c: float, m: float) -> float:
    """∂L_R/∂(θ_c0 + θ_p0) при θ = 0, без штрафа."""
    v = torch.zeros(RL.n_par("CK"), dtype=torch.float64, requires_grad=True)
    RL.loss_r(v, V, "CK", 0.0, c, m, "R").backward()
    return float(v.grad[2] + v.grad[3])
