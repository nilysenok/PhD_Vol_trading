"""Проверки S-RCD-IMPL-2 (`rd_walk`): --check-nest (W.G2: Н1, Н2, Н3), --check (W.G4; sha256 выходов до и после), далее `rd_walk_pit` (--check-pit W.G3, --check-level W.G5). Только чтение."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as CM
from models_v2.s18 import cm_walk as CW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_walk as RD
from models_v2.s18.r_walk_ctx import make_ctx

ROOT = RD.RG.ROOT
RCK, CD, CK = (ROOT / "data/regen/harj_struct" / p for p in ("R/rck.parquet", "CM/cd.parquet", "CM/ck.parquet"))
ACT = {"rcd": ("2020-08-03", 75), "cdf": ("2018-07-02", 100), "ckf": ("2018-07-02", 100)}
_CTX: dict = {}


def ctx_of(level: str = "F"):
    if level not in _CTX:
        _CTX[level] = make_ctx("CK", level)
    return _CTX[level]


def zero_parts(var: str, ctx) -> list:
    c8 = ctx.x["C"].to_numpy(float)
    return [(a, b, RD.zero_month_sc(np.arange(0), c8[a:b])) for a, b in W.month_bounds(ctx)]


def nest_h1() -> list:
    bad, ctx, w5 = [], ctx_of(), pd.read_parquet(RL.W5G)
    for var in RD.VARS:
        df = RD.assemble(var, ctx, zero_parts(var, ctx))
        same = [np.array_equal(df["C"].to_numpy(), ctx.x["C"].to_numpy(float)), np.array_equal(df["F"].to_numpy(), w5["F"].to_numpy()), np.array_equal(df["X1_R"].to_numpy(), ctx.x["F"].to_numpy(float))]
        print("Н1 %s: θ ≡ 0 во всех месяцах: C = cfg8, F = W5-G, X1_R = X1 побитно: %s" % (var, same))
        bad += ["Н1 %s %s" % (var, same)] * (not all(same))
    return bad


def nest_h2() -> list:
    ctx = ctx_of()
    df = RD.assemble("rcd", ctx, RD.walk("rcd", ctx, spec=("CK", "R")), spec=("CK", "R"))
    ref = pd.read_parquet(RCK)
    cols = ["C", "F", "X1_R", "lam", "c", "m"] + [c for c in ref.columns if c.startswith(("w_", "G_")) or c in CM.NAMES["CK"]]
    try:
        pd.testing.assert_frame_equal(df[cols].reset_index(drop=True), ref[cols].reset_index(drop=True), check_exact=True)
        ok = True
    except AssertionError as e:
        ok = False
        print(str(e)[:400])
    print("Н2: проход R с формой CK = R/rck.parquet побитно (%d колонок): %s" % (len(cols), ok))
    return ["Н2: проход R с формой CK ≠ rck.parquet"] * (not ok)


def nest_h3() -> list:
    bad, ctx = [], ctx_of("C")
    for var, ref_p in (("cdf", CD), ("ckf", CK)):
        df, ref = RD.assemble(var, ctx, RD.walk(var, ctx, level="C")), pd.read_parquet(ref_p)
        both = (df["lam"].to_numpy() == ref["lam"].to_numpy()) | (df["lam"].isna().to_numpy() & ref["lam"].isna().to_numpy())
        xd, mb = pd.DatetimeIndex(df["d"]), YA.month_starts(pd.DatetimeIndex(df["d"]))
        nl = sum(bool(both[a]) for a in mb if xd[a] >= YA.ACTIVATION)
        nm = sum(1 for a in mb if xd[a] >= YA.ACTIVATION)
        rc = float(np.max(np.abs(df["C"].to_numpy() / ref["C"].to_numpy() - 1.0)))
        q = lambda f: float(RL.ell(ref["rv"].to_numpy(), np.log(f)).mean())
        rq = abs(q(df["F"].to_numpy()) - q(ref["F"].to_numpy())) / q(ref["F"].to_numpy())
        print("Н3 %s (уровень C) против %s: λ совпал в %d из %d активных месяцев; max отн. |C − C_сохр| = %.2e (допуск 1e-4); |ΔQLIKE(F)|/QLIKE = %.2e (допуск 1e-5)" % (var, ref_p.name, nl, nm, rc, rq))
        bad += ["Н3 %s: λ %d из %d — СТОП и разбор" % (var, nl, nm)] * (nl != nm) + ["Н3 %s: C %.2e" % (var, rc)] * (rc > 1e-4) + ["Н3 %s: QLIKE %.2e" % (var, rq)] * (rq > 1e-5)
    return bad


def check_nest() -> list:
    bad = nest_h1() + nest_h2()
    return bad + nest_h3() if not bad else bad


def tree(d) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.parquet"))} if d.is_dir() else {}


def check_file(var: str, df: pd.DataFrame) -> list:
    form, proto = RD.VARS[var]
    ctx, first, n_act = ctx_of(), *ACT[var]
    xd, mb = pd.DatetimeIndex(ctx.x["d"]), W.month_bounds(ctx)
    act, bad = df["active"].to_numpy(), []
    th = df[list(CM.NAMES[form])].to_numpy()
    mon = [(a, b) for a, b in mb if act[a]]
    ok = len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == xd).all() and len(mon) == n_act and str(xd[mon[0][0]].date()) == first
    lo, hi = np.array(CM.bounds(form)).T
    ok = ok and (th >= lo).all() and (th <= hi).all() and np.array_equal(df["C"].to_numpy()[~act], df["cfg8"].to_numpy()[~act])
    lams = {RD.W.LAMS[0], RD.W.LAMS[1], RD.W.LAMS[2]} | {0.0} if proto == "R" else set(CM.LAMS)
    ok = ok and set(df["lam"][act]) <= lams and df["lam"][~act].isna().all()
    D = ctx.D
    for a, b in mon:     # C = cfg8·𝓜(θ): SC — побитно, R — exp(ln C) (допуск 1e-12, ↺ путь exp(mean ln C) как в r_walk)
        V = D.view(np.arange(a, b))
        c_chk = D.c8[a:b] * CM.forward(torch.as_tensor(th[a]), V, form)["M"].numpy()
        c = df["C"].to_numpy()[a:b]
        if df["lam"].iloc[a] == 0.0 and proto == "R":
            ok = ok and np.array_equal(c, df["cfg8"].to_numpy()[a:b])
        else:
            ok = ok and (np.array_equal(c, c_chk) if proto == "SC" else float(np.max(np.abs(c / c_chk - 1.0))) <= 1e-12)
    if proto == "R":
        ref = pd.read_parquet(RCK)
        ok = ok and all(np.array_equal(df[k].to_numpy(), ref[k].to_numpy(), equal_nan=True) for k in ("c", "m"))
        for a, b in mon:
            G = {lam: [df["G_%s_b%d" % (k, i)].iloc[a] for i in (1, 2, 3) if not np.isnan(df["G_%s_b%d" % (k, i)].iloc[a])] for lam, k in zip(W.LAMS, W.LK)}
            lam, w = W.decide(G, "blocks")
            ok = ok and lam == df["lam"].iloc[a] and tuple(w[k] for k in W.LAMS) == tuple(df["w_" + k].iloc[a] for k in W.LK) and bool((th[a] == 0).all()) == (lam == 0.0)
        print("%s: активных месяцев %d (с %s), λ* %s, θ ≠ 0 в %d из %d" % (var, len(mon), first, sorted(set(df["lam"][act])), sum(bool((th[a] != 0).any()) for a, _ in mon), len(mon)))
    else:
        ok = ok and (df["c"][act] == np.inf).all() and (df["m"][act] == 1.0).all() and df["lam"][act].ne(0.0).all()
        small = [a for a, _ in mon if df["n_train"].iloc[a] < CW.MIN_VALID]
        ok = ok and len(small) == 6 and all(df["lam"].iloc[a] == CW.LAM_DEF for a in small)
        for a, _ in mon:
            if a in small:
                continue
            v = [df["val_" + k].iloc[a] for k in RD.VK]
            ok = ok and df["lam"].iloc[a] == max(l for l, q in zip(CM.LAMS, v) if q == min(v))
        print("%s: активных месяцев %d (с %s), λ-малые %d, λ %s; ‖θ‖₀ по месяцам: %s" % (var, len(mon), first, len(small), {l: int((df["lam"][act] == l).sum() > 0) for l in CM.LAMS}, [int((th[a] != 0).sum()) for a, _ in mon]))
    return ["%s: состав выхода" % var] * (not ok)


def check() -> list:
    h0, bad = tree(RD.RD_DIR), []
    for v, f in RD.OUTS.items():
        p = RD.RD_DIR / f
        bad += ["нет %s" % f] if not p.exists() else check_file(v, pd.read_parquet(p))
    h1 = tree(RD.RD_DIR)
    print("sha256 выходов до/после равны: %s (%d файлов)" % (h0 == h1, len(h1)))
    return bad + ([] if h0 == h1 else ["--check изменил выходы"])


def main(args: list) -> int:
    from models_v2.s18 import rd_walk_pit as P
    fn = {"--check-nest": check_nest, "--check": check, "--check-pit": P.check_pit, "--check-level": P.check_level}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --run VAR, --check-nest, --check-pit, --check-level или --check")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
