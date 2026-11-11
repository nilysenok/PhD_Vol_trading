"""Проверки S-Z-IMPL-1 (`z_model`): --check-nest (Z1.G1), --check-grad (Z1.G2), --check-pit (Z1.G3), --synth (Z1.G4). Только чтение."""
from __future__ import annotations

import torch  # noqa: F401  (первым, класс № 152)

import copy
import hashlib
import math

import numpy as np
import pandas as pd

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_feat as F
from models_v2.s18 import z_model as ZM

ROOT, _C = F.ROOT, {}
SHA = {"models_v2/s18/r_loss.py": "3ae5c9b1af7e38e4", "models_v2/s18/r_walk.py": "9021113db1d3a0bb", "models_v2/s18/r_walk_ctx.py": "8262991b3bdf70f5", "models_v2/s18/cm_model.py": "a3f8d71d553e4aeb",
       "models_v2/s18/cm_layers.py": "1d8e0b7e498eeb10", "models_v2/s18/cm_walk.py": "96d0a3bf002cbc01", "models_v2/s17/fwd_z.py": "d4dfb8e7fac5084e", "models_v2/s17/fwd_store.py": "c6e0deb859dc03d9",
       "models_v2/s17/fwd_src.py": "b577f68a2f7b9f50", "models_v2/s14/v7_1b_sta.py": "71dae6fd4f2ef26d", "models_v2/s14/v7_1e_rep.py": "de3d660dc47fb835", "data/regen/harj_struct/R/rck.parquet": "97d6fff21834f09e",
       "data/regen/harj_struct/CM/feat.parquet": "e31691e0d0dfff1f", "data/regen/harj_x1/W5/w5g.parquet": "0ae40e1bcc7691e1", "data/regen/harj_deep/E1/x1.parquet": "56d9351607780959",
       "data/fwd/models/new/x1.parquet": "12c2b3614fd2a63b"}


def sha_bad() -> list:
    bad = ["sha %s" % p for p, h in SHA.items() if not hashlib.sha256((ROOT / p).read_bytes()).hexdigest().startswith(h)]
    print("sha256 входов (%d файлов) = таблице спецификации: %s" % (len(SHA), "да" if not bad else bad))
    return bad


def real() -> tuple:
    if not _C:
        _C.update(ctx=RC.make_ctx("CK"), zm=F.zmaps())
    return _C["ctx"], _C["zm"]


def setup(ctx, zm, date: str, s: int = 0) -> tuple:
    xd = pd.DatetimeIndex(ctx.x["d"])
    a = int(np.flatnonzero(xd >= pd.Timestamp(date))[0])
    P = RL.pool_of("CK", ctx.D, ctx.x, a)
    z = zm[xd[a].year][s]
    return ctx.view(P), ZM.ztil(z, *ZM.std_stats(z, P))[P], RL.clip_stats(ctx.x0, P), P, a


def eq(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def check_nest() -> list:
    bad, (ctx, zm), rng = sha_bad(), real(), np.random.default_rng(0)
    bnd = np.array(CM.bounds("CK"))
    ths, nd = [bnd[:, 0] + (bnd[:, 1] - bnd[:, 0]) * rng.random(6) for _ in range(3)], dict(h1=0, rz=0, same=0, zero=0, form=0, mut=0, mean=0, k=0)
    for y in F.YEARS:
        for s in F.SEEDS:
            z, rows = zm[y][s], np.flatnonzero(np.isfinite(zm[y][s]).all(1))
            V, zt = ctx.view(rows), ZM.ztil(z, *ZM.std_stats(z, rows))[rows]
            zts = [ZM.ztil(zm[y][q], *ZM.std_stats(zm[y][q], rows))[rows] for q in F.SEEDS]
            nd["rz"] += not (eq(RL.ln_f(ZM.lnc_head(torch.zeros(4), V, zt, "Z"), V).numpy(), ctx.lnf0[rows]) and eq(ZM.lnc_head(torch.zeros(4), V, zt, "Z").numpy(), RL.lnc_of(torch.zeros(6, dtype=torch.float64), V, "CK").numpy()))
            for th in ths:
                t, ref = torch.as_tensor(th), torch.log(V.c8 * CM.forward(torch.as_tensor(th), V, "CK")["M"]).numpy()
                full, thz = np.concatenate([th, np.zeros(4)]), 0.3 * rng.normal(size=4)
                l0 = ZM.lnc_head(full, V, zt, "CKZ").numpy()
                nd["h1"] += not eq(l0, ref)
                nd["same"] += not eq(ZM.predict_lnc([full] * 5, V, [zt] * 5), l0)
                nd["zero"] += not eq(ZM.predict_lnc([full] * 5, V, zts), l0)        # θ_Z = 0: головы различаются только z̃ — среднее отклонений ровно 0
                M = CM.forward(t, V, "CK")["M"]
                lz, eta = ZM.lnc_head(np.concatenate([th, thz]), V, zt, "CKZ"), ZM.eta_of(zt, torch.as_tensor(thz))
                nd["form"] += not (eq(lz.numpy(), (torch.log(V.c8 * M) + eta).numpy()) and np.allclose(lz.numpy(), np.log((V.c8 * M).numpy()) + zt @ thz, rtol=1e-12, atol=0))
                nd["mut"] += not eq(torch.log(V.c8 * M * torch.exp(eta)).numpy(), lz.numpy())     # мутант «η внутри логарифма»: сколько (энкодер × θ) он расходится побитно
                nd["mean"] += not eq(np.mean([l0] * 5, axis=0), l0)                              # мутант «np.mean»
                nd["k"] += 1
    print("Н1: ln C_Z (θ_Z = 0) = log(c8·M_CK) побитно: расхождений %d из %d (45 энкодеров × 3 θ_CK, 2198 дат у Y = 2026)" % (nd["h1"], nd["k"]))
    print("S-RZ при θ_Z = 0: ln C = log(c8) = `lnc_of(0)` и ln F = Ctx.lnf0 побитно: расхождений %d из 45" % nd["rz"])
    print("среднее одинаковых голов = голова 0 побитно: расхождений %d; θ_Z = 0 и разные z̃ у голов: %d из %d" % (nd["same"], nd["zero"], nd["k"]))
    print("θ_Z ≠ 0: ln C = log(c8·M) + η побитно к независимому пересчёту: расхождений %d" % nd["form"])
    print("мутанты (информация): η внутри логарифма расходится побитно в %d из %d; np.mean вместо «головы 0 + отклонений» — в %d из %d" % (nd["mut"], nd["k"], nd["mean"], nd["k"]))
    return bad + ["%s: %d" % (k, nd[k]) for k in ("h1", "rz", "same", "zero", "form") if nd[k]]


def check_grad() -> list:
    bad, (ctx, zm), rng = [], real(), np.random.default_rng(1)
    V, zt, (c, m), P, a = setup(ctx, zm, "2023-02-01")
    n_clip = int((np.abs(zt) >= ZM.CLIP - 1e-12).sum())
    print("месяц %s: |P| = %d, c = %.3f, m = %.4f; значений z̃ на границе ±%g: %d из %d" % (pd.DatetimeIndex(ctx.x["d"])[a].date(), len(P), c, m, ZM.CLIP, n_clip, zt.size))
    bad += ["нет обрезанных z̃"] * (n_clip < 1)
    for var in ("CKZ", "Z"):
        pts = [np.concatenate([CM._x0("CK", r), 0.3 * rng.normal(size=4)]) if var == "CKZ" else 0.3 * rng.normal(size=4) for r in range(1, 6)]
        hi = []
        for x_ in pts:
            f = lambda v: ZM.loss_z(v, V, zt, var, 1e-2, c, m, "R")
            v = torch.as_tensor(x_).clone().requires_grad_(True)
            f(v).backward()
            fd = np.array([(float(f(torch.as_tensor(x_ + h))) - float(f(torch.as_tensor(x_ - h)))) / 2e-6 for h in 1e-6 * np.eye(len(x_))])
            g = v.grad.numpy()
            err = float(np.max(np.abs(g - fd)) / max(np.max(np.abs(g)), 1e-8))
            u = (torch.log(V.rv) - RL.ln_f(ZM.lnc_head(torch.as_tensor(x_), V, zt, var), V) - math.log(m)).numpy()
            hi.append(int((u > math.log(c / m)).sum()))
            bad += ["градиент %s: %.2e" % (var, err)] * (err > 1e-6)
            print("  %s: отн. расхождение autograd / разностей = %.2e; пар с u > ln κ: %d" % (var, err, hi[-1]))
        bad += ["%s: < 2 точек с u > ln κ" % var] * (sum(k > 0 for k in hi) < 2)
    return bad


def corrupt(ctx, z: np.ndarray, T) -> tuple:
    """Всё вне T: RV ×7, π и βg сдвинуты, z → 7z + 1000."""
    out = np.ones(len(z), bool)
    out[T] = False
    Dc, rv = copy.copy(ctx.D), np.array(ctx.D.rv, float)
    rv[out] *= 7.0
    Dc.rv = rv
    lv = ctx.lv
    lc = RL.Lv(np.where(out, 0.3, lv.pi), lv.lnb, np.where(out, lv.bg + 0.5, lv.bg), lv.has)
    return Dc, lc, np.where(out[:, None], 7.0 * z + 1e3, z)


def check_pit() -> list:
    bad, (ctx, zm), n = [], real(), len(real()[0].x)
    act = [a for a, _ in RW.month_bounds(ctx) if len(RL.pool_of("CK", ctx.D, ctx.x, a)) >= RW.MIN_POOL]
    th = torch.as_tensor(np.concatenate([CM._x0("CK", 2), [0.2, -0.1, 0.3, 0.1]]))
    for a in np.random.default_rng(0).choice(act, 3, replace=False):
        d0, P = pd.DatetimeIndex(ctx.x["d"])[a], RL.pool_of("CK", ctx.D, ctx.x, a)
        z = zm[d0.year][0]
        for tag, T in [("месяц D", P)] + [("блок V_%d" % (i + 1), Tb) for i, (Tb, _) in enumerate(RW.blocks(P))]:
            Dc, lc, zc = corrupt(ctx, z, T)
            x1 = RL.x0_of(RL.view(Dc, lc, np.arange(n)), "CK")
            s0, s1 = ZM.std_stats(z, T), ZM.std_stats(zc, T)
            l0 = float(ZM.loss_z(th, ctx.view(T), ZM.ztil(z, *s0)[T], "CKZ", 1e-2, *RL.clip_stats(ctx.x0, T), "R"))
            l1 = float(ZM.loss_z(th, RL.view(Dc, lc, T), ZM.ztil(zc, *s1)[T], "CKZ", 1e-2, *RL.clip_stats(x1, T), "R"))
            ok = eq(s0[0], s1[0]) and eq(s0[1], s1[1]) and RL.clip_stats(ctx.x0, T) == RL.clip_stats(x1, T) and l0 == l1
            W = np.flatnonzero(np.isfinite(z).all(1))     # мутант: вся выборка года энкодера (включая дни после D)
            w0, w1 = ZM.std_stats(z, W), ZM.std_stats(zc, W)
            print("  %s %s: μ, σ, c, m, L_R %s; мутант «μ, σ по всей выборке» %s" % (tag, d0.date(), "не изменились" if ok else "ИЗМЕНИЛИСЬ", "изменился (ловится)" if not eq(w0[0], w1[0]) else "НЕ ловится"))
            bad += ["PIT %s %s" % (tag, d0.date())] * (not ok) + ["мутант не ловится"] * eq(w0[0], w1[0])
    return bad


def synth_pair(zt, seed: int, sig_z: float, pi: float = 1.0):
    from models_v2.s18.r_loss_syn import synth_data
    D, lv = synth_data(len(zt), seed, np.zeros(6), pi=pi)
    D.rv = D.rv * np.exp(sig_z * zt[:, 0])
    return D, lv, RL.view(D, lv, np.arange(len(zt)))


def fit(D, lv, V, zt, lam: float, rows=None) -> np.ndarray:
    r = np.arange(len(zt)) if rows is None else rows
    return ZM.fit_head(V, zt, "CKZ", lam, *RL.clip_stats(RL.x0_of(V, "CK"), r), "R")


def oos(V, zt, th) -> float:
    return float(RL.ell(V.rv.numpy(), RL.ln_f(ZM.lnc_head(torch.as_tensor(th), V, zt, "CKZ"), V).numpy()).mean())


def null_part(z, n: int, lam: float) -> tuple:
    """(а): n первых дат z; среднее по 20 повторам (QLIKE вне выборки R − θ=0, отн.), его ошибка среднего, доля θ ≠ 0."""
    zt, r = ZM.ztil(z[:n], *ZM.std_stats(z[:n], np.arange(n))), []
    for s in range(20):
        th, Vt = fit(*synth_pair(zt, s, 0.0), zt, lam), synth_pair(zt, 1000 + s, 0.0)[2]
        r.append((oos(Vt, zt, th) / oos(Vt, zt, np.zeros(10)) - 1.0, float(np.abs(th).max() > 1e-6)))
    a = np.array(r)
    return a[:, 0].mean(), a[:, 0].std(ddof=1) / math.sqrt(20), a[:, 1].mean()


def synth() -> list:
    bad, (_, zm), z = [], real(), real()[1][2026][0]
    n = len(z)
    mu, sd = ZM.std_stats(z, np.arange(n))
    zt = ZM.ztil(z, mu, sd)
    print("z̃ (энкодер 2026, seed 0, все %d дат D_M): доля обрезанных %.2f %%" % (n, 100 * float((np.abs(ZM.ztil_raw(z, mu, sd)) > ZM.CLIP).mean())))
    for nn, gate in ((n, True), (2000, False)):     # (а): критерий — на всех датах D_M при λ = 0,1 (как `r_loss_syn`); n = 2000 первых дат и другие λ — печать (чувствительность)
        for lam in RL.GRID:
            e, se, sh = null_part(z, nn, lam)
            print("(а) нулевой сигнал, n = %d%s, λ = %g: QLIKE вне выборки R − θ=0 = %+.4f %% (± %.4f), доля θ ≠ 0 = %.2f" % (nn, "" if gate else " (чувствительность)", lam, 100 * e, 100 * se, sh))
            bad += ["(а) λ = 0,1: %+.4f %%" % (100 * e)] * (gate and lam == RL.GRID[0] and e > 1e-3)
    est = np.array([fit(*synth_pair(zt, 500 + s, 0.2), zt, 1e-3) for s in range(20)])[:, 6]
    print("(б) ln RV = ln F⁰ + 0,2·z̃₀ + ошибка (π = 1), λ = 1e-3: θ̂_Z,0 > 0 в %d из 20; медиана %.3f; |медиана − 0,2| = %.3f" % ((est > 0).sum(), np.median(est), abs(np.median(est) - 0.2)))
    bad += ["(б) знак %d из 20" % (est > 0).sum()] * ((est > 0).sum() < 18) + ["(б) |медиана − 0,2| = %.3f" % abs(np.median(est) - 0.2)] * (abs(np.median(est) - 0.2) > 0.1)
    D, lv, V = synth_pair(zt, 7, 0.2)
    F0 = np.exp(RL.ln_f(RL.lnc_of(torch.zeros(6, dtype=torch.float64), V, "CK"), V).numpy())
    D.rv = D.rv.copy()
    D.rv[0] = F0[0] * math.exp(0.5)     # пара 0: RV не связан с z; сдвигается только её z (z̃₀ = 30 до обрезки при μ, σ без выброса)
    V = RL.view(D, lv, np.arange(n))
    z30 = z.copy()
    z30[0, 0] = mu[0] + 30.0 * sd[0]
    base, clip, raw = fit(D, lv, V, zt, 1e-3), fit(D, lv, V, ZM.ztil(z30, mu, sd), 1e-3), fit(D, lv, V, ZM.ztil_raw(z30, mu, sd), 1e-3)
    dc, dr = np.linalg.norm(clip - base), np.linalg.norm(raw - base)
    print("(в) пара z̃₀ = 30: сдвиг θ̂ с обрезкой = %.5f, без обрезки = %.5f, отношение %.3f" % (dc, dr, dc / dr))
    return bad + ["(в) отношение %.3f" % (dc / dr)] * (dc > 0.2 * dr)


def main(args: list) -> int:
    fn = {"--check-nest": check_nest, "--check-grad": check_grad, "--check-pit": check_pit, "--synth": synth}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --check-nest, --check-grad, --check-pit или --synth")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
