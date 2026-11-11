"""Проверки S-R-IMPL-2 (`r_walk`): --check-nest (W.G2), --check-pit (W.G3), --check (W.G4; sha256 выходов до и после). Только чтение."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s17.fwd_src import LAST_SESSION, Src
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_feat_chk as FC
from models_v2.s18 import a_model as AM
from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_model as CM
from models_v2.s18 import cm_walk_chk as CWC
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W

ACT = {"A": ("2019-08-01", 87), "CK": ("2020-08-03", 75)}
_CTX: dict = {}


def ctx_of(arch: str) -> W.Ctx:
    if arch not in _CTX:
        _CTX[arch] = W.make_ctx(arch)
    return _CTX[arch]


def zero_walk(ctx: W.Ctx) -> pd.DataFrame:
    """θ = 0 во всех месяцах: пустой пул ⇒ C_R = cfg8 побитно, слои X1 и W5 над ним."""
    c8 = ctx.x["C"].to_numpy(float)
    parts = [(a, b, W.fit_month(ctx, np.arange(0), ctx.dp[a:b], c8[a:b], "R")) for a, b in W.month_bounds(ctx)]
    return W.assemble("ra2" if ctx.arch == "A" else "rck", ctx, parts)


def check_nest() -> list:
    bad, w5 = [], pd.read_parquet(RL.W5G)
    for arch in ("A", "CK"):
        ctx = ctx_of(arch)
        df = zero_walk(ctx)
        e1, e2 = float(np.max(np.abs(df["F"] / w5["F"] - 1.0))), float(np.max(np.abs(df["X1_R"] / ctx.x["F"].to_numpy(float) - 1.0)))
        same = np.array_equal(df["C"].to_numpy(), ctx.x["C"].to_numpy(float))
        print("%s: θ = 0 во всех месяцах: max отн. |F_R − W5-G| = %.2e, |X1_R − X1| = %.2e; C_R = cfg8 побитно: %s" % (arch, e1, e2, same))
        bad += ["%s: вложение %.2e / %.2e / %s" % (arch, e1, e2, same)] * (max(e1, e2) > 1e-12 or not same)
    return bad


def _same(a: dict, b: dict, keys) -> bool:
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in keys)


def _spoiled(arch: str, ctx: W.Ctx, D0, d_last, seeds) -> tuple:
    """(контекст с порчей данных после D0, контекст с порчей после d_last и RV пар t⁺ > D0)."""
    x, tp = ctx.x, pd.DatetimeIndex(ctx.D.tp if arch == "A" else ctx.x["t_plus"])
    mask = np.asarray(tp > D0)
    if arch == "A":
        src = Src()
        bars, idays = src.bars("SBER", LAST_SESSION), src.index_days(LAST_SESSION)
        k1, cfg, bmix = AM.load_inputs()
        mk = lambda d: AM.build_data(A.build(FC.corrupt_day(bars, d, seeds), idays), k1, cfg, bmix)
        Da, Db = mk(D0), mk(d_last)
        Db.rv_next = np.where(mask, Db.rv_next * 7.0, Db.rv_next)
    else:
        tab, x2 = CF.load_c(), x.copy()
        x2.loc[x2.index[mask], "rv"] = np.random.default_rng(1).uniform(1.0, 1e3, int(mask.sum()))
        Da, Db = CM.build_data(x, CF.build(tab=CWC._spoil(tab, D0, 2))), CM.build_data(x2, CF.build(tab=CWC._spoil(tab, d_last, 3)))
    return W.Ctx(arch, Da, ctx.lv, x, ctx.gap, ctx.dp), W.Ctx(arch, Db, ctx.lv, x, ctx.gap, ctx.dp)


def check_pit(n_months: int = 3, seed: int = 0, archs=("A", "CK")) -> list:
    bad, rng = [], np.random.default_rng(seed)
    for arch in archs:
        ctx = ctx_of(arch)
        xd, mb = pd.DatetimeIndex(ctx.x["d"]), W.month_bounds(ctx)
        act = [(a, b) for a, b in mb if xd[a] >= pd.Timestamp(ACT[arch][0]) and b < len(xd)]
        for k in rng.choice(len(act) - 1, n_months, replace=False):
            a, b = act[int(k)]
            ca, cb = _spoiled(arch, ctx, xd[a], xd[b - 1], int(rng.integers(1_000_000)))
            res = [W.fit_month(c, RL.pool_of(arch, c.D, c.x, a), c.dp[a:b], c.x["C"].to_numpy(float)[a:b], "R") for c in (ctx, ca, cb)]
            ok_a, ok_b = _same(res[0], res[1], ("lam", "w", "G", "c", "m", "theta")), _same(res[0], res[2], ("lam", "w", "G", "c", "m", "theta", "C"))
            print("PIT %s месяц D = %s: λ*, w, G, c, m, θ при порче > D: %s; плюс прогноз при порче > конца месяца и RV пар t⁺ > D: %s; λ* = %s" % (arch, xd[a].date(), ok_a, ok_b, res[0]["lam"]))
            bad += ["порча будущего изменила переоценку %s %s" % (arch, xd[a].date())] * (not (ok_a and ok_b))
    return bad


def tree(d) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.parquet"))} if d.is_dir() else {}


def derived(df: pd.DataFrame, rule: str) -> list:
    """λ* и w, выведенные из колонок G по правилу разд. 2.4 для каждого активного месяца (по refit_R)."""
    bad = []
    for _, g in df[df["active"]].groupby("refit_R"):
        r = g.iloc[0]
        G = {lam: [r["G_%s_b%d" % (k, b)] for b in (1, 2, 3) if not np.isnan(r["G_%s_b%d" % (k, b)])] for lam, k in zip(W.LAMS, W.LK)}
        lam, w = W.decide(G, rule)
        pars = g[[c for c in df.columns if c in set(W.names("A")) | set(W.names("CK"))]]
        bad += ["λ* ≠ правилу %s" % pd.Timestamp(r["refit_R"]).date()] * (lam != r["lam"])
        bad += ["w ≠ правилу %s" % pd.Timestamp(r["refit_R"]).date()] * (tuple(w[k] for k in W.LAMS) != tuple(r["w_" + k] for k in W.LK))
        bad += ["θ ≡ 0 ⇔ λ* = 0 нарушено %s" % pd.Timestamp(r["refit_R"]).date()] * (bool((pars.to_numpy() == 0).all()) != (r["lam"] == 0.0))
    return bad


def check_file(var: str, df: pd.DataFrame, x) -> list:
    arch, mode, rule = W.VARS[var]
    first, n_act = ACT[arch]
    bad, act = [], df[df["active"]]
    ok = len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == pd.DatetimeIndex(x["d"])).all()
    nm = act["refit_R"].nunique()
    ok = ok and nm == n_act and str(pd.Timestamp(act["refit_R"].min()).date()) == first
    ok = ok and set(act["lam"].unique()) <= {0.0, *W.LAMS} and df.loc[~df["active"], "lam"].isna().all()
    ok = ok and np.isfinite(df[["F", "X1_R", "C"]].to_numpy()).all() and (df[["F", "X1_R"]] > 0).all().all()
    pre = df["active"].to_numpy() == 0
    ok = ok and np.array_equal(df["C"].to_numpy()[pre], df["cfg8"].to_numpy()[pre])
    zero = df["active"].to_numpy() & (df["lam"].to_numpy() == 0.0)
    ok = ok and np.array_equal(df["C"].to_numpy()[zero], df["cfg8"].to_numpy()[zero])
    if mode == "plain":
        ok = ok and (act["m"] == 1.0).all() and np.isinf(act["c"]).all()
    else:
        ok = ok and np.isfinite(act["c"]).all() and (act["c"] > act["m"]).all()
    b3 = [c for c in df.columns if c.endswith("_b3")]
    early = [c for c in df.columns if c.endswith(("_b1", "_b2"))]
    ok = ok and (act[b3].notna().all().all()) and (act[early].notna().all().all() if rule == "blocks" else act[early].isna().all().all())
    print("%s: строк %d, активных месяцев %d (с %s), λ* %s: %s" % (var, len(df), nm, pd.Timestamp(act["refit_R"].min()).date(), sorted(set(act["lam"])), "ok" if ok else "FAIL"))
    return (["%s: состав выхода" % var] * (not ok)) + derived(df, rule)


def repro(var: str, df: pd.DataFrame) -> list:
    arch, mode, rule = W.VARS[var]
    ctx = ctx_of(arch)
    xd = pd.DatetimeIndex(ctx.x["d"])
    a, b = [(a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp("2022-04-01")][0]
    r = W.fit_month(ctx, RL.pool_of(arch, ctx.D, ctx.x, a), ctx.dp[a:b], ctx.x["C"].to_numpy(float)[a:b], mode, rule)
    err = float(np.max(np.abs(r["C"] / df["C"].to_numpy()[a:b] - 1.0)))
    print("пересчёт месяца %s, %s: max отн. |C/parquet − 1| = %.2e" % (pd.Timestamp(ctx.x["d"].iloc[a]).date(), var, err))
    return ["%s: пересчёт месяца ≠ parquet" % var] * (err > 1e-10)


def check() -> list:
    h0, bad, outs = tree(W.R_DIR), [], {}
    x = C.load()
    for v, f in W.OUTS.items():
        p = W.R_DIR / f
        if not p.exists():
            bad.append("нет %s" % f)
            continue
        outs[v] = pd.read_parquet(p)
        bad += check_file(v, outs[v], x) + repro(v, outs[v])
    for a, ra, rb, rc in (("A", "ra2", "ra2b", "ra2c"), ("CK", "rck", "rckb", "rckc")):
        if all(k in outs for k in (ra, rb, rc)):
            same = all(np.array_equal(outs[ra]["active"], outs[k]["active"]) and np.array_equal(outs[ra]["refit_R"].dropna(), outs[k]["refit_R"].dropna()) for k in (rb, rc))
            print("%s: включение и даты переоценки у R, Rb, Rc совпадают: %s" % (a, same))
            bad += ["%s: разное включение" % a] * (not same)
    h1 = tree(W.R_DIR)
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])
