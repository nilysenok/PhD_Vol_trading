"""Проверки S-D-IMPL-1/2 (`d_walk`): --check-default (D1.G3), --check-nest (D1.G4), --check-pit (D1.G5), --check (D2.G2; sha256 выходов до и после). Только чтение."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import d_walk as D
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as RC

_CTX: dict = {}
ACT = RC.ACT["CK"]


def ctx_of(var: str) -> W.Ctx:
    if var not in _CTX:
        _CTX[var] = W.make_ctx("CK", D.LEVEL_D[var])
    return _CTX[var]


def check_default() -> list:
    bad, ctx = [], W.make_ctx("CK")
    ok1 = ctx.x0_train is ctx.x0 and ctx.lv_train is None
    mb = W.month_bounds(ctx)
    pools = [RL.pool_of("CK", ctx.D, ctx.x, mb[i][0]) for i in (40, 65, 90)]
    ok2 = all(torch.equal(getattr(ctx.view_train(P), k), getattr(ctx.view(P), k)) for P in pools for k in ("l_pi", "l_lnb", "l_bg", "rv", "c8"))
    try:
        W.make_ctx("A", "C")
        ok3 = False
    except ValueError:
        ok3 = True
    cc = W.make_ctx("CK", "C")
    P = pools[-1]
    ok4 = bool((cc.view_train(P).l_pi == 1).all() and (cc.view_train(P).l_lnb == 0).all() and (cc.view_train(P).l_bg == 0).all())
    ok4 = ok4 and torch.equal(cc.view(P).l_pi, ctx.view(P).l_pi) and np.array_equal(cc.lnf0, ctx.lnf0) and not (cc.x0_train is cc.x0)
    print("умолчание: x0_train is x0: %s; view_train == view на %d пулах: %s; make_ctx('A', 'C') → ValueError: %s; уровень C: обучение π = 1, ln B = 0, βg = 0, валидация и lnf0 прежние: %s" % (ok1, len(pools), ok2, ok3, ok4))
    return ["умолчания Ctx ≠ прежнему поведению"] * (not (ok1 and ok2 and ok3 and ok4))


def forced_zero(var: str) -> pd.DataFrame:
    """λ* = 0 принудительно во всех активных месяцах (choose подменён; пул не пуст — c, m считаются, θ = 0)."""
    ctx, orig = ctx_of(var), W.choose
    W.choose = lambda ctx_, P, mode, rule="blocks": (0.0, {lam: 0 for lam in W.LAMS}, {lam: [0.0] * W.NB for lam in W.LAMS})
    try:
        c8, mode = ctx.x["C"].to_numpy(float), D.VARS_D[var][1]
        parts = [(a, b, W.fit_month(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], mode)) for a, b in W.month_bounds(ctx)]
        return W.assemble(var, ctx, parts)
    finally:
        W.choose = orig


def check_nest() -> list:
    bad, w5, x1 = [], pd.read_parquet(RL.W5G), C.load()
    for var in D.VARS_D:
        ctx, df = ctx_of(var), forced_zero(var)
        e1, e2 = float(np.max(np.abs(df["F"] / w5["F"] - 1.0))), float(np.max(np.abs(df["X1_R"] / x1["F"].to_numpy(float) - 1.0)))
        same = np.array_equal(df["C"].to_numpy(), ctx.x["C"].to_numpy(float))
        print("%s: λ* = 0 принудительно (%d активных дней): C = cfg8 побитно: %s; max отн. |F − W5-G| = %.2e, |X1_R − X1| = %.2e" % (var, int(df["active"].sum()), same, e1, e2))
        bad += ["%s: вложение %.2e / %.2e / %s" % (var, e1, e2, same)] * (max(e1, e2) > 1e-12 or not same or not df["active"].any())
    return bad


def _lv_ctx(var: str, c: W.Ctx) -> W.Ctx:
    return W.Ctx(c.arch, c.D, c.lv, c.x, c.gap, c.dp, lv_train=ctx_of(var).lv_train)


def check_pit(n_months: int = 3, seed: int = 0) -> list:
    bad, rng = [], np.random.default_rng(seed)
    for var, (_, mode, rule) in D.VARS_D.items():
        ctx = ctx_of(var)
        xd, mb = pd.DatetimeIndex(ctx.x["d"]), W.month_bounds(ctx)
        act = [(a, b) for a, b in mb if xd[a] >= pd.Timestamp(ACT[0]) and b < len(xd)]
        for k in rng.choice(len(act) - 1, n_months, replace=False):
            a, b = act[int(k)]
            ca, cb = (_lv_ctx(var, c) for c in RC._spoiled("CK", ctx, xd[a], xd[b - 1], int(rng.integers(1_000_000))))
            res = [W.fit_month(c, RL.pool_of("CK", c.D, c.x, a), c.dp[a:b], c.x["C"].to_numpy(float)[a:b], mode, rule) for c in (ctx, ca, cb)]
            ok_a, ok_b = RC._same(res[0], res[1], ("lam", "w", "G", "c", "m", "theta")), RC._same(res[0], res[2], ("lam", "w", "G", "c", "m", "theta", "C"))
            print("PIT %s месяц D = %s: λ*, w, G, c, m, θ при порче > D: %s; прогноз при порче > конца месяца и RV пар t⁺ > D: %s; λ* = %s" % (var, xd[a].date(), ok_a, ok_b, res[0]["lam"]))
            bad += ["порча будущего изменила переоценку %s %s" % (var, xd[a].date())] * (not (ok_a and ok_b))
    return bad


def check_file(var: str, df: pd.DataFrame, ctx: W.Ctx) -> list:
    _, mode, rule = D.VARS_D[var]
    x, act = ctx.x, df[df["active"]]
    ok = len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == pd.DatetimeIndex(x["d"])).all()
    nm = act["refit_R"].nunique()
    ok = ok and nm == ACT[1] and str(pd.Timestamp(act["refit_R"].min()).date()) == ACT[0]
    ok = ok and set(act["lam"].unique()) <= {0.0, *W.LAMS} and df.loc[~df["active"], "lam"].isna().all()
    ok = ok and np.isfinite(df[["F", "X1_R", "C"]].to_numpy()).all() and (df[["F", "X1_R"]] > 0).all().all()
    c8, cc = df["cfg8"].to_numpy(), df["C"].to_numpy()
    zero = ~df["active"].to_numpy() | (df["lam"].to_numpy() == 0.0)
    ok = ok and np.array_equal(cc[zero], c8[zero])
    ok = ok and np.isfinite(act["c"]).all() and (act["c"] > act["m"]).all() and (var != "rckm" or (act["m"] == 1.0).all())
    ok = ok and act[[c for c in df.columns if c.endswith("_b3")]].notna().all().all() and act[[c for c in df.columns if c.endswith(("_b1", "_b2"))]].notna().all().all()
    if var == "rckf":    # опорное отношение уровня C — независимо от Ctx: RV / cfg8
        ok = ok and np.allclose(ctx.x0_train, x["rv"].to_numpy(float) / x["C"].to_numpy(float), rtol=1e-12, atol=0)
    mon = [a for a, _ in W.month_bounds(ctx) if df["active"].to_numpy()[a]]
    cm = np.asarray([W.stats_of(ctx.x0_train, RL.pool_of("CK", ctx.D, x, a), mode) for a in mon])
    ok = ok and len(mon) == ACT[1] and np.array_equal(cm, df[["c", "m"]].to_numpy()[mon])
    print("%s: строк %d, активных месяцев %d (с %s), λ* %s; c, m = clip_stats(%s, P) во всех месяцах: %s" % (var, len(df), nm, pd.Timestamp(act["refit_R"].min()).date(), sorted(set(act["lam"])), "RV/cfg8" if var == "rckf" else "RV/W5-G", "ok" if ok else "FAIL"))
    return (["%s: состав выхода" % var] * (not ok)) + RC.derived(df, rule)


def repro(var: str, df: pd.DataFrame, k: int = 3) -> list:
    ctx, bad = ctx_of(var), []
    xd = pd.DatetimeIndex(ctx.x["d"])
    act = [(a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp(ACT[0])]
    for a, b in [act[i] for i in np.linspace(0, len(act) - 1, k).astype(int)]:
        r = W.fit_month(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], ctx.x["C"].to_numpy(float)[a:b], D.VARS_D[var][1], "blocks")
        same = np.array_equal(r["C"], df["C"].to_numpy()[a:b])
        print("пересчёт %s месяц %s: C побитно = parquet: %s (max отн. %.2e)" % (var, xd[a].date(), same, float(np.max(np.abs(r["C"] / df["C"].to_numpy()[a:b] - 1.0)))))
        bad += ["%s: пересчёт месяца %s ≠ parquet" % (var, xd[a].date())] * (not same)
    return bad


def check() -> list:
    h0, bad = RC.tree(D.D_DIR), []
    for var, f in D.OUTS_D.items():
        if not (D.D_DIR / f).exists():
            bad.append("нет %s" % f)
            continue
        df = pd.read_parquet(D.D_DIR / f)
        bad += check_file(var, df, ctx_of(var)) + repro(var, df)
    h1 = RC.tree(D.D_DIR)
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])


if __name__ == "__main__":
    fs = {"--check-default": check_default, "--check-nest": check_nest, "--check-pit": check_pit, "--check": check}
    k = [a for a in sys.argv if a in fs]
    if not k:
        raise SystemExit("укажите " + ", ".join(fs))
    b = fs[k[0]]()
    print("%s: %s" % (k[0], "ok" if not b else "FAIL " + "; ".join(b)))
    sys.exit(1 if b else 0)
