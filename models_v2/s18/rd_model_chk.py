"""Проверки S-RCD-IMPL-1 (`rd_model`): --check-nest (M.G1), --check-grad (M.G2), --check-pit (M.G3), --synth (M.G4). Только чтение."""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import rd_model as RM

ROOT = C.ROOT
SHA = {"models_v2/s18/r_loss.py": "3ae5c9b1af7e38e4", "models_v2/s18/r_walk.py": "9021113db1d3a0bb", "models_v2/s18/r_walk_ctx.py": "8262991b3bdf70f5", "models_v2/s18/cm_model.py": "a3f8d71d553e4aeb",
       "models_v2/s18/cm_walk.py": "96d0a3bf002cbc01", "models_v2/s18/cm_layers.py": "1d8e0b7e498eeb10", "models_v2/s18/cm_feat.py": "093be85e40a9cd68", "models_v2/s18/a_eval_stat.py": "74fb64773aafeb27",
       "models_v2/s18/a_eval_spa.py": "b941131469704d0c", "data/regen/harj_struct/CM/cd.parquet": "7bd9f5690950cda4", "data/regen/harj_struct/CM/ck.parquet": "979e71ad28ba16b9",
       "data/regen/harj_struct/CM/feat.parquet": "e31691e0d0dfff1f", "data/regen/harj_struct/R/rck.parquet": "97d6fff21834f09e", "data/regen/harj_struct/R/rckb.parquet": "e35ebf7248958bfa",
       "data/regen/harj_struct/D/rckf.parquet": "52740db6f9811952", "data/regen/harj_struct/Z/rckz.parquet": "e8284586b668126e", "data/regen/harj_struct/L/rl1.parquet": "28bddc1668d50f6c",
       "data/regen/harj_x1/W5/w5g.parquet": "0ae40e1bcc7691e1", "data/regen/harj_deep/E1/x1.parquet": "56d9351607780959"}
MONTHS = ("2021-03-01", "2023-02-01", "2025-06-01")
_CACHE: dict = {}


def sha_bad() -> list:
    return ["sha %s" % p for p, h in SHA.items() if not hashlib.sha256((ROOT / p).read_bytes()).hexdigest().startswith(h)]


def load() -> tuple:
    if "d" not in _CACHE:
        from models_v2.s18 import cm_walk as W
        x, _, D = W.inputs("CK")
        _CACHE["d"] = (D, x, RL.level_arrays("CK", D, x))
    return _CACHE["d"]


def months(x) -> list:
    xd = pd.DatetimeIndex(x["d"])
    return [int(np.flatnonzero(xd >= pd.Timestamp(s))[0]) for s in MONTHS]


def zero(form: str):
    return torch.zeros(CM.NP, dtype=torch.float64)


def x0_of(V, form: str) -> np.ndarray:
    return (V.rv * torch.exp(-RL.ln_f(RM.lnc_of(zero(form), V, form), V))).numpy()


def check_nest() -> list:
    bad, (D, x, lv) = sha_bad(), load()
    n = len(lv.pi)
    V = RL.view(D, lv, np.arange(n))
    a, b, r = (RM.lnc_of(zero("CD"), V, "CD"), RM.lnc_of(zero("CK"), V, "CK"), RL.lnc_of(zero("CK"), V, "CK"))
    ok = torch.equal(a, b) and torch.equal(b, r)
    print("θ = 0: ln C (CD) = ln C (CK) = r_loss.lnc_of на %d датах побитно: %s" % (n, ok))
    bad += ["ln C при θ = 0 (CD/CK/RL) не совпали"] * (not ok)
    rng, P = np.random.default_rng(0), RL.pool_of("CK", D, x, months(x)[1])
    VP, x0 = RL.view(D, lv, P), RL.x0_of(V, "CK")
    c, m = RL.clip_stats(x0, P)
    for i in range(5):
        lo, hi = np.array(CM.bounds("CK")).T
        v = torch.as_tensor(np.clip(rng.normal(0.0, 0.4, 6), lo, hi))
        for mode in RM.MODES:
            same = torch.equal(RM.loss(v, VP, "CK", 1e-2, c, m, mode), RL.loss_r(v, VP, "CK", 1e-2, c, m, mode))
            bad += ["loss(CK, %s) ≠ loss_r, θ №%d" % (mode, i)] * (not same)
    print("loss(·, CK) = RL.loss_r побитно в 5 случайных θ в границах, оба режима: %s" % (not any("loss(CK" in s for s in bad)))
    for a_ in months(x):
        P = RL.pool_of("CK", D, x, a_)
        VP, (c, m) = RL.view(D, lv, P), RL.clip_stats(x0, P)
        t_new, t_old = RM.fit(VP, "CK", 1e-2, c, m, "R")[0], RL.fit_r(VP, "CK", 1e-2, c, m, "R")[0]
        same = np.array_equal(t_new, t_old)
        print("  fit(CK) = fit_r(CK) побитно, месяц %s, |P| = %d: %s" % (pd.DatetimeIndex(x["d"])[a_].date(), len(P), same))
        bad += ["fit(CK) ≠ fit_r %s" % a_] * (not same)
        for form in RM.FORMS:
            s = torch.zeros((), dtype=torch.float64, requires_grad=True)
            v = torch.zeros(CM.NP, dtype=torch.float64, requires_grad=True)
            RM.loss(v, VP, form, 0.0, c, m, "R", shift=s).backward()
            print("  %s: балл сдвига ln F при θ = 0 = %.2e" % (form, float(s.grad)))
            bad += ["балл уровня %s %.2e" % (form, float(s.grad))] * (abs(float(s.grad)) > 1e-12)
    th = torch.as_tensor(RM.starts("CD", (3,))[0])
    pl = RM.loss(th, VP, "CD", 0.0, float("inf"), 1.0, "plain")
    ref = RL.ell(VP.rv.numpy(), RL.ln_f(RM.lnc_of(th, VP, "CD"), VP).numpy()).mean()
    bad += ["plain ≠ mean ℓ"] * (abs(float(pl) - ref) > 1e-12)
    return bad


def check_grad() -> list:
    bad, (D, x, lv) = [], load()
    P = RL.pool_of("CK", D, x, months(x)[1])
    V, x0 = RL.view(D, lv, P), RL.x0_of(RL.view(D, lv, np.arange(len(lv.pi))), "CK")
    c, m = RL.clip_stats(x0, P)
    for form in RM.FORMS:
        pts = RM.starts(form, (1, 2, 3, 4)) + [np.array([0.3, 1.0, 0.0, -0.2, 0.4, 0.1])]
        n_hi, bnd = [], 0
        for p in pts:
            f = lambda v: RM.loss(v, V, form, 1e-2, c, m, "R")
            v = torch.as_tensor(p).clone().requires_grad_(True)
            f(v).backward()
            fd = np.array([(float(f(torch.as_tensor(p + h))) - float(f(torch.as_tensor(p - h)))) / 2e-6 for h in 1e-6 * np.eye(6)])
            g = v.grad.numpy()
            err = float(np.max(np.abs(g - fd)) / max(np.max(np.abs(g)), 1e-8))
            u = (torch.log(V.rv) - RL.ln_f(RM.lnc_of(torch.as_tensor(p), V, form), V) - math.log(m)).numpy()
            n_hi.append(int((u > math.log(c / m)).sum()))
            bnd += int(any(p[i] in (0.0, 1.0) for i in CM.UNIT[form]))
            bad += ["градиент %s: %.2e" % (form, err)] * (err > 1e-6)
            print("  %s: отн. расхождение autograd / разностей = %.2e; пар с u > ln κ: %d" % (form, err, n_hi[-1]))
        bad += ["%s: < 2 точек с u > ln κ" % form] * (sum(k > 0 for k in n_hi) < 2)
        bad += ["%s: нет точки с ρ / γ на границе" % form] * (bnd < 1)
    return bad


def corrupt(D, lv, mask) -> tuple:
    Dc = copy.copy(D)
    for k, f in (("rv", 7.0), ("c8", 2.0), ("Hc", 3.0), ("Hp", 3.0), ("su", 0.5), ("labc", 4.0), ("labp", 4.0)):
        a = np.array(getattr(D, k), float)
        a[mask] *= f
        setattr(Dc, k, a)
    for k, val in (("sc", 0.3), ("sp", 0.2)):
        a = np.array(getattr(D, k), float)
        a[mask] = val
        setattr(Dc, k, a)
    for k in ("zc", "zp"):
        a = np.array(getattr(D, k), float)
        a[mask] += 1.0
        setattr(Dc, k, a)
    return Dc, RL.Lv(np.where(mask, 0.3, lv.pi), lv.lnb, np.where(mask, lv.bg + 0.5, lv.bg), lv.has)


def check_pit() -> list:
    bad, (D, x, lv) = [], load()
    n = len(lv.pi)
    for form in RM.FORMS:
        v = torch.as_tensor(RM.starts(form, (2,))[0])
        x0 = x0_of(RL.view(D, lv, np.arange(n)), form)
        for a in months(x):
            P = RL.pool_of("CK", D, x, a)
            for tag, T in (("месяц D", P), ("блок V_3", P[:-126])):
                mask = np.ones(n, bool)
                mask[T] = False
                Dc, lc = corrupt(D, lv, mask)
                x1 = x0_of(RL.view(Dc, lc, np.arange(n)), form)
                s0, s1 = RL.clip_stats(x0, T), RL.clip_stats(x1, T)
                l0, l1 = (float(RM.loss(v, RL.view(d_, l_, T), form, 1e-2, *s, "R")) for d_, l_, s in ((D, lv, s0), (Dc, lc, s1)))
                ok = s0 == s1 and l0 == l1
                print("  %s %s %s: c, m, L_R %s" % (form, tag, pd.DatetimeIndex(x["d"])[a].date(), "не изменились" if ok else "ИЗМЕНИЛИСЬ"))
                bad += ["PIT %s %s" % (form, tag)] * (not ok)
    return bad


def main(args: list) -> int:
    from models_v2.s18.rd_model_syn import synth
    fn = {"--check-nest": check_nest, "--check-grad": check_grad, "--check-pit": check_pit, "--synth": synth}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --check-nest, --check-grad, --check-pit или --synth")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
