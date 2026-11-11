"""Проверки S-L-IMPL-1/2 (`l_walk`): --check-nest (L1.G2), --check-pit (L1.G3), --check-equiv (L1.G4), --check-convex (L1.G5), --check-ctrl (L1.G6), --check (L2.G2, L2.G3; sha256 выходов до и после). Только чтение."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import l_chk_aux as AUX
from models_v2.s18 import l_model as ML
from models_v2.s18 import l_walk as LW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as RC

_CTX: dict = {}
ACT = RC.ACT["CK"]
KEYS = ("lam", "w", "G", "c", "m", "a")


def ctx_of() -> W.Ctx:
    if "ctx" not in _CTX:
        _CTX["ctx"] = W.make_ctx("CK")
    return _CTX["ctx"]


def forced_zero(kind: str) -> pd.DataFrame:
    """λ* = 0 принудительно во всех активных месяцах (choose_l подменён; пул не пуст — c, m считаются, a = 0)."""
    ctx, orig = ctx_of(), LW.choose_l
    LW.choose_l = lambda ctx_, P, k: (0.0, {lam: 0 for lam in W.LAMS}, {lam: [0.0] * W.NB for lam in W.LAMS})
    try:
        c8 = ctx.x["C"].to_numpy(float)
        return LW.assemble_l(ctx, [(a, b, LW.fit_month_l(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], kind)) for a, b in W.month_bounds(ctx)])
    finally:
        LW.choose_l = orig


def check_nest() -> list:
    bad, w5, x1, ctx = [], pd.read_parquet(RL.W5G), C.load(), ctx_of()
    for var, kind in LW.VARS_L.items():
        df = forced_zero(kind)
        e1, e2 = float(np.max(np.abs(df["F"] / w5["F"] - 1.0))), float(np.max(np.abs(df["X1_R"] / x1["F"].to_numpy(float) - 1.0)))
        same = np.array_equal(df["C"].to_numpy(), ctx.x["C"].to_numpy(float))
        print("%s: λ* = 0 принудительно (%d активных дней): C = cfg8 побитно: %s; max отн. |F − W5-G| = %.2e, |X1_R − X1| = %.2e" % (var, int(df["active"].sum()), same, e1, e2))
        bad += ["%s: вложение %.2e / %.2e / %s" % (var, e1, e2, same)] * (max(e1, e2) > 1e-12 or not same or not df["active"].any())
    return bad


def check_pit(n_months: int = 3, seed: int = 0) -> list:
    bad, ctx = [], ctx_of()
    xd = pd.DatetimeIndex(ctx.x["d"])
    rng = np.random.default_rng(seed)
    act = [(a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp(ACT[0]) and b < len(xd)]
    for k in rng.choice(len(act) - 1, n_months, replace=False):
        a, b = act[int(k)]
        ca, cb = RC._spoiled("CK", ctx, xd[a], xd[b - 1], int(rng.integers(1_000_000)))
        for var, kind in LW.VARS_L.items():
            res = [LW.fit_month_l(c, RL.pool_of("CK", c.D, c.x, a), c.dp[a:b], c.x["C"].to_numpy(float)[a:b], kind) for c in (ctx, ca, cb)]
            ok_a, ok_b = RC._same(res[0], res[1], KEYS), RC._same(res[0], res[2], KEYS + ("C",))
            print("PIT %s месяц D = %s: λ*, w, G, c, m, â при порче > D: %s; прогноз при порче > конца месяца и RV пар t⁺ > D: %s; λ* = %s" % (var, xd[a].date(), ok_a, ok_b, res[0]["lam"]))
            bad += ["порча будущего изменила переоценку %s %s" % (var, xd[a].date())] * (not (ok_a and ok_b))
    return bad


def derived(df: pd.DataFrame) -> list:
    bad = []
    for _, g in df[df["active"]].groupby("refit_R"):
        r, d = g.iloc[0], pd.Timestamp(g["refit_R"].iloc[0]).date()
        G = {lam: [r["G_%s_b%d" % (k, b)] for b in (1, 2, 3)] for lam, k in zip(W.LAMS, W.LK)}
        lam, w = W.decide(G, "blocks")
        bad += ["λ* ≠ правилу %s" % d] * (lam != r["lam"])
        bad += ["w ≠ правилу %s" % d] * (tuple(w[k] for k in W.LAMS) != tuple(r["w_" + k] for k in W.LK))
    return bad


def check_file(var: str, df: pd.DataFrame, ctx: W.Ctx) -> list:
    kind, x, act = LW.VARS_L[var], ctx.x, df[df["active"]]
    ok = len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == pd.DatetimeIndex(x["d"])).all()
    nm = act["refit_R"].nunique()
    ok = ok and nm == ACT[1] and str(pd.Timestamp(act["refit_R"].min()).date()) == ACT[0]
    ok = ok and set(act["lam"].unique()) <= {0.0, *W.LAMS} and df.loc[~df["active"], "lam"].isna().all()
    ok = ok and np.isfinite(df[["F", "X1_R", "C"]].to_numpy()).all() and (df[["F", "X1_R"]] > 0).all().all()
    a, zero = df["a"].to_numpy(), ~df["active"].to_numpy() | (df["lam"].to_numpy() == 0.0)
    ok = ok and (np.abs(a) <= ML.A_BOUND).all() and (a[zero] == 0.0).all() and (a[~zero] != 0.0).all()
    c8, cc = df["cfg8"].to_numpy(), df["C"].to_numpy()
    ok = ok and np.array_equal(cc[zero], c8[zero]) and np.array_equal(cc[~zero], (c8 * np.exp(a))[~zero])
    A, mon = LW.arrays(ctx), [i for i, _ in W.month_bounds(ctx) if df["active"].to_numpy()[i]]
    cm = np.asarray([ML.stats_l(kind, A, RL.pool_of("CK", ctx.D, x, i)) for i in mon])
    ok = ok and len(mon) == ACT[1] and np.array_equal(cm, df[["c", "m"]].to_numpy()[mon]) and (act["c"] > act["m"]).all()
    b3, early = [c for c in df.columns if c.endswith("_b3")], [c for c in df.columns if c.endswith(("_b1", "_b2"))]
    ok = ok and act[b3 + early].notna().all().all()
    print("%s: строк %d, активных месяцев %d (с %s), λ* %s, |â| ≤ %.2f, a ≡ 0 ⇔ λ* = 0: %s" % (var, len(df), nm, pd.Timestamp(act["refit_R"].min()).date(), sorted(set(act["lam"])), ML.A_BOUND, "ok" if ok else "FAIL"))
    bad = ["%s: состав выхода" % var] * (not ok) + derived(df)
    if var == "rl1a":
        same = np.array_equal(df["F"].to_numpy(), pd.read_parquet(RL.W5G)["F"].to_numpy())
        print("rl1a: F = w5g.F побитно во всех 2198 днях: %s" % same)
        bad += ["rl1a: F ≠ w5g.F побитно (P-L3)"] * (not same)
    return bad


def repro(var: str, df: pd.DataFrame, k: int = 3) -> list:
    ctx, bad = ctx_of(), []
    for a, b in AUX.active_months(ctx, k):
        r = LW.fit_month_l(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], ctx.x["C"].to_numpy(float)[a:b], LW.VARS_L[var])
        same = np.array_equal(r["C"], df["C"].to_numpy()[a:b]) and np.all(df["a"].to_numpy()[a:b] == r["a"])
        print("пересчёт %s месяц %s: C и a побитно = parquet: %s" % (var, pd.Timestamp(ctx.x["d"].iloc[a]).date(), same))
        bad += ["%s: пересчёт месяца %s ≠ parquet" % (var, pd.Timestamp(ctx.x["d"].iloc[a]).date())] * (not same)
    return bad


def check() -> list:
    h0, bad = RC.tree(LW.L_DIR), []
    for var, f in LW.OUTS_L.items():
        if not (LW.L_DIR / f).exists():
            bad.append("нет %s" % f)
            continue
        df = pd.read_parquet(LW.L_DIR / f)
        bad += check_file(var, df, ctx_of()) + repro(var, df)
    h1 = RC.tree(LW.L_DIR)
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])


if __name__ == "__main__":
    fs = {"--check-nest": check_nest, "--check-pit": check_pit, "--check-equiv": lambda: AUX.check_equiv(ctx_of()), "--check-convex": lambda: AUX.check_convex(ctx_of()),
          "--check-ctrl": lambda: AUX.check_ctrl(ctx_of()), "--check": check}
    k = [a for a in sys.argv if a in fs]
    if not k:
        raise SystemExit("укажите " + ", ".join(fs))
    b = fs[k[0]]()
    print("%s: %s" % (k[0], "ok" if not b else "FAIL " + "; ".join(b)))
    sys.exit(1 if b else 0)
