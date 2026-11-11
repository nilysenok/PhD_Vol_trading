"""Проверки S-Z-IMPL-2 (`z_walk`): --check-nest (Z2.G2), --check-pit (Z2.G3), --check (Z2.G4; sha256 выходов до и после), --synth-null (Z2.G7). Только чтение."""
from __future__ import annotations

import torch  # noqa: F401  (первым, класс № 152)

import hashlib

import numpy as np
import pandas as pd

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_chk as RWC
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_feat as ZF
from models_v2.s18 import z_reg as ZR
from models_v2.s18 import z_walk as ZW

KEYS = ("C", "F", "X1_R", "lam", "w_1e-1", "w_1e-2", "w_1e-3", "c", "m", *[f"G_{k}_b{b}" for k in RW.LK for b in (1, 2, 3)], *RW.names("CK"))
FIRST, N_ACT = "2020-08-03", 75


def nest_rck(ctx, zm, rck) -> list:
    df = ZW.assemble(ctx, ZW.walk(ctx, zm, "CKZ", False, quiet=True))
    try:
        pd.testing.assert_frame_equal(df[list(KEYS)], rck[list(KEYS)], check_exact=True)
        ok = True
    except AssertionError as e:
        ok = False
        print(str(e)[:300])
    print("Н2 S-RCKZ use_z=False против rck.parquet (C, F, X1_R, lam, w, G, c, m, θ_CK) побитно: %s; активных месяцев %d" % (ok, df.loc[df["active"], "refit_R"].nunique()))
    return [] if ok else ["Н2 rckz ≠ rck"]


def nest_rz(ctx, zm) -> list:
    df, w5 = ZW.assemble(ctx, ZW.walk(ctx, zm, "Z", False, quiet=True)), pd.read_parquet(RL.W5G)
    lam0 = bool((df.loc[df["active"], "lam"] == 0.0).all()) and df["active"].sum() > 0
    same = np.array_equal(df["F"].to_numpy(), w5["F"].to_numpy()) and np.array_equal(df["C"].to_numpy(), df["cfg8"].to_numpy())
    print("Н2 S-RZ use_z=False: λ* = 0 во всех активных месяцах: %s; F = W5-G и C = cfg8 побитно: %s" % (lam0, same))
    return [] if lam0 and same else ["Н2 rz: λ0 %s, F = W5-G %s" % (lam0, same)]


def diag_h3(ctx, zm, rck) -> None:
    df = ZW.assemble(ctx, ZW.walk(ctx, zm, "CKZ", True, zero=True, quiet=True))
    e = lambda k: float(np.max(np.abs(df[k].to_numpy() / rck[k].to_numpy() - 1.0)))
    print("Н3 (печать, не гейт) use_z=True и z̃ ≡ 0 против rck.parquet: max отн. отклонение C = %.2e, F = %.2e, X1_R = %.2e" % (e("C"), e("F"), e("X1_R")))


def check_nest() -> list:
    ctx, (zm, _), rck = RC.make_ctx("CK"), ZW.load_z(), pd.read_parquet(ZW.RCK)
    bad = nest_rck(ctx, zm, rck) + nest_rz(ctx, zm)
    diag_h3(ctx, zm, rck)
    return bad


def _zcorrupt(zsrc: dict, mask) -> dict:
    return {y: {s: np.where(mask[:, None], 7.0 * z + 1e3, z) for s, z in d.items()} for y, d in zsrc.items()}


def _res(c, zsrc, a, b, var, shift) -> dict:
    Y = pd.Timestamp(c.x["d"].iloc[a]).year
    return ZW.fit_month(c, [zsrc[Y][s] for s in ZW.SEEDS], RL.pool_of("CK", c.D, c.x, a), c.dp[a:b], c.x["C"].to_numpy(float)[a:b], var)


def check_pit(n_months: int = 3, seed: int = 0) -> list:
    bad, rng, ctx, (zm, zs) = [], np.random.default_rng(seed), RC.make_ctx("CK"), ZW.load_z()
    xd, mb = pd.DatetimeIndex(ctx.x["d"]), RW.month_bounds(ctx)
    act = [(a, b) for a, b in mb if xd[a] >= pd.Timestamp(FIRST) and b < len(xd)]
    for k in rng.choice(len(act) - 1, n_months, replace=False):
        a, b = act[int(k)]
        ca, cb = RWC._spoiled("CK", ctx, xd[a], xd[b - 1], int(rng.integers(1_000_000)))
        for key, (var, shift) in ZW.VARS.items():
            zsrc = zs if shift else zm
            r0, r1, r2 = _res(ctx, zsrc, a, b, var, shift), _res(ca, _zcorrupt(zsrc, np.asarray(xd > xd[a])), a, b, var, shift), _res(cb, _zcorrupt(zsrc, np.asarray(xd > xd[b - 1])), a, b, var, shift)
            ok_a, ok_b = RWC._same(r0, r1, ("lam", "w", "G", "c", "m", "theta", "thz")), RWC._same(r0, r2, ("lam", "w", "G", "c", "m", "theta", "thz", "C"))
            print("PIT %s месяц D = %s: λ*, w, G, c, m, θ при порче данных, z и строк C > D: %s; плюс прогноз при порче > конца месяца и RV пар t⁺ > D: %s; λ* = %s" % (key, xd[a].date(), ok_a, ok_b, r0["lam"]))
            bad += ["порча будущего изменила переоценку %s %s" % (key, xd[a].date())] * (not (ok_a and ok_b))
    return bad


def tree(d) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*d.glob("*.parquet"), *d.glob("zmaps.npz")])} if d.is_dir() else {}


def derived(df: pd.DataFrame) -> list:
    bad, pc = [], RW.names("CK") + ["thz_%d" % j for j in range(4)]
    for _, g in df[df["active"]].groupby("refit_R"):
        r = g.iloc[0]
        G = {lam: [r["G_%s_b%d" % (k, b)] for b in (1, 2, 3) if not np.isnan(r["G_%s_b%d" % (k, b)])] for lam, k in zip(RW.LAMS, RW.LK)}
        lam, w = RW.decide(G, "blocks")
        d0 = pd.Timestamp(r["refit_R"]).date()
        bad += ["λ* ≠ правилу %s" % d0] * bool(lam != r["lam"])
        bad += ["w ≠ правилу %s" % d0] * (tuple(w[k] for k in RW.LAMS) != tuple(r["w_" + k] for k in RW.LK))
        bad += ["θ ≡ 0 ⇔ λ* = 0 нарушено %s" % d0] * (bool((g[pc].to_numpy() == 0).all()) != bool(r["lam"] == 0.0))
    return bad


def check_file(key: str, df: pd.DataFrame, x, rck) -> list:
    act, ok = df[df["active"]], len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == pd.DatetimeIndex(x["d"])).all()
    nm = act["refit_R"].nunique()
    ok = ok and nm == N_ACT and str(pd.Timestamp(act["refit_R"].min()).date()) == FIRST and set(act["lam"].unique()) <= {0.0, *RW.LAMS} and df.loc[~df["active"], "lam"].isna().all()
    ok = ok and np.isfinite(df[["F", "X1_R", "C"]].to_numpy()).all() and (df[["F", "X1_R", "C"]] > 0).all().all()
    pre = df["active"].to_numpy() == 0
    zero = df["active"].to_numpy() & (df["lam"].to_numpy() == 0.0)
    ok = ok and np.array_equal(df["C"].to_numpy()[pre | zero], df["cfg8"].to_numpy()[pre | zero]) and np.isfinite(act["c"]).all() and (act["c"] > act["m"]).all()
    cm = key == "rz" or key == "rckz"
    same_cm = np.array_equal(df[["c", "m"]].to_numpy(), rck[["c", "m"]].to_numpy(), equal_nan=True)
    print("%s: строк %d, активных месяцев %d (с %s), λ* %s: %s; c, m = rck.parquet побитно: %s; n_excl > 0: %d дат; z_ok = 0: %d дат; n_clip (среднее по активным) %.4f" % (
        key, len(df), nm, pd.Timestamp(act["refit_R"].min()).date(), sorted(set(act["lam"])), "ok" if ok else "FAIL", same_cm, int((df["n_excl"] > 0).sum()), int((df["z_ok"] == 0).sum()), float(act["n_clip"].mean())))
    return ["%s: состав выхода" % key] * (not ok) + ["%s: c, m ≠ rck" % key] * (cm and not same_cm) + ["%s: n_excl или z_ok" % key] * bool((df["n_excl"] > 0).any() or (df["z_ok"] == 0).any()) + derived(df)


def repro(key: str, df: pd.DataFrame, ctx, zm, zs, rck) -> list:
    var, shift = ZW.VARS[key]
    xd = pd.DatetimeIndex(ctx.x["d"])
    a, b = [(a, b) for a, b in RW.month_bounds(ctx) if xd[a] >= pd.Timestamp("2022-04-01")][0]
    Y = xd[a].year
    r = ZW.fit_month(ctx, [(zs if shift else zm)[Y][s] for s in ZW.SEEDS], RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], ctx.x["C"].to_numpy(float)[a:b], var, True, rck["C"].to_numpy(float)[a:b] if var == "CKZ" else None)
    err = float(np.max(np.abs(r["C"] / df["C"].to_numpy()[a:b] - 1.0)))
    print("пересчёт месяца %s, %s: max отн. |C/parquet − 1| = %.2e" % (xd[a].date(), key, err))
    return ["%s: пересчёт месяца ≠ parquet" % key] * (err > 1e-10)


def check() -> list:
    h0, bad, ctx, (zm, zs) = tree(ZW.Z_DIR), [], RC.make_ctx("CK"), ZW.load_z()
    rck = pd.read_parquet(ZW.RCK)
    for k, f in ZW.OUTS.items():
        if not (ZW.Z_DIR / f).exists():
            bad.append("нет %s" % f)
            continue
        df = pd.read_parquet(ZW.Z_DIR / f)
        bad += check_file(k, df, ctx.x, rck) + repro(k, df, ctx, zm, zs, rck)
    bad += ZR.check()
    h1 = tree(ZW.Z_DIR)
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])


def main(args: list) -> int:
    from models_v2.s18 import z_walk_syn as SY
    fn = {"--check-nest": check_nest, "--check-pit": check_pit, "--check": check, "--synth-null": SY.synth_null}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --check-nest, --check-pit, --check или --synth-null")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
