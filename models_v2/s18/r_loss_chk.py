"""Проверки S-R-IMPL-1 (`r_loss`): --check-nest (L.G1), --check-grad (L.G2), --check-pit (L.G3), --synth (L.G4). Только чтение."""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import a_eval as AE
from models_v2.s18 import a_model as AM
from models_v2.s18 import cm_eval as CE
from models_v2.s18 import r_loss as RL

ROOT = C.ROOT
R_SHA = {"models_v2/s18/a_model.py": "36d9003cf999b05d", "models_v2/s18/a_feat.py": "4ea866e6d691c7c8", "models_v2/s18/a_walk.py": "2535a2950ec28a65", "models_v2/s18/a_layers.py": "fbaf8f18f6e57e45",
         "models_v2/s18/cm_model.py": "a3f8d71d553e4aeb", "models_v2/s18/cm_feat.py": "093be85e40a9cd68", "models_v2/s18/cm_walk.py": "96d0a3bf002cbc01", "models_v2/s18/cm_layers.py": "1d8e0b7e498eeb10",
         "data/regen/harj_struct/A/a2.parquet": "b67eacad25acef73", "data/regen/harj_struct/CM/ck.parquet": "979e71ad28ba16b9", "data/regen/harj_struct/CM/feat.parquet": "e31691e0d0dfff1f",
         "data/regen/harj_x1/W5/w5g.parquet": "0ae40e1bcc7691e1", "data/regen/harj_x1/W8/w8g.parquet": "1c91df7fab8b4edd", "docs/otchety/harj_struct/S_B_DIAG.csv": "bbd0492097b81ff3"}
MONTHS = ("2021-03-01", "2023-02-01", "2025-06-01")
_CACHE: dict = {}


def sha_bad() -> list:
    want = {**AE.SPEC_SHA, **CE.C_SHA, **R_SHA}
    return ["sha %s" % p for p, h in want.items() if not hashlib.sha256((ROOT / p).read_bytes()).hexdigest().startswith(h[:16])]


def load(arch: str) -> tuple:
    if arch not in _CACHE:
        if arch == "A":
            D, x = AM.load_data(), C.load()
        else:
            from models_v2.s18 import cm_walk as W
            x, _, D = W.inputs("CK")
        _CACHE[arch] = (D, x, RL.level_arrays(arch, D, x))
    return _CACHE[arch]


def months(x) -> list:
    xd = pd.DatetimeIndex(x["d"])
    return [int(np.flatnonzero(xd >= pd.Timestamp(s))[0]) for s in MONTHS]


def known_level(fn) -> list:
    """Известный ответ: π, βg по дню d пары (не t⁺), βg только при gap = 1 и refit_date ≠ NaT."""
    d = pd.date_range("2020-01-01", periods=6)
    x = pd.DataFrame(dict(d=d, pi=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], M=np.arange(1.0, 7.0)))
    w5 = pd.DataFrame(dict(d=d, gap=[1, 1, 0, 1, 0, 1], beta=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], refit_date=[pd.NaT, d[1], d[2], d[3], d[4], d[5]]))
    lv = fn("CK", x=x, w5=w5)
    return (["π не дня d"] * (not np.array_equal(lv.pi, x["pi"].to_numpy())) + ["βg"] * (not np.array_equal(lv.bg, [0.0, 0.6, 0.0, 0.8, 0.0, 1.0]))
            + ["ln B"] * (not np.allclose(lv.lnb, np.log(x["M"]))))


def check_rho() -> list:
    bad, kap = [], 3.5
    u, lk = torch.linspace(-3.0, 1.2, 200, dtype=torch.float64), math.log(3.5)
    f = 0.9
    ref = RL.ell(np.exp(u.numpy()) * f, np.full(200, math.log(f)))
    e = float(np.max(np.abs(RL.rho(u, kap).numpy() - ref)))
    bad += [] if e <= 1e-12 else ["ρ ≠ ℓ(RV/m, F) при u ≤ ln κ: %.2e" % e]
    a, b = torch.tensor(lk, dtype=torch.float64, requires_grad=True), torch.tensor(np.nextafter(lk, 9.0), dtype=torch.float64, requires_grad=True)
    ra, rb = RL.rho(a, kap), RL.rho(b, kap)
    ra.backward(), rb.backward()
    e = max(abs(float(ra - rb)), abs(float(a.grad - b.grad)), abs(float(a.grad) - (kap - 1.0)))
    print("ρ: u ≤ ln κ max|ρ − ℓ| = %.2e; разрыв значения и производной в ln κ = %.2e" % (float(np.max(np.abs(RL.rho(u, kap).numpy() - ref))), e))
    return bad + ([] if e <= 1e-12 else ["ρ или производная разрывны в ln κ: %.2e" % e])


def check_nest() -> list:
    bad, w5 = sha_bad(), pd.read_parquet(RL.W5G)
    for arch in ("A", "CK"):
        D, x, lv = load(arch)
        V = RL.view(D, lv, np.arange(len(lv.pi)))
        F0 = V.rv.numpy() / RL.x0_of(V, arch)
        has = lv.has > 0
        pos = pd.Series(np.arange(len(x)), index=pd.DatetimeIndex(x["d"])).reindex(pd.DatetimeIndex(D.d if arch == "A" else x["d"])).to_numpy()
        e1 = float(np.max(np.abs(F0[has] / w5["F"].to_numpy()[pos[has].astype(int)] - 1.0)))
        e2 = 0.0 if has.all() else float(np.max(np.abs(F0[~has] / np.exp(D.lnk1[~has]) - 1.0)))
        print("%s: θ = 0: max отн. |F − w5g.F| = %.2e (%d пар); у пар без X1 |F − K1| = %.2e (%d пар)" % (arch, e1, has.sum(), e2, (~has).sum()))
        bad += ["%s: вложение %.2e / %.2e" % (arch, e1, e2)] * (max(e1, e2) > 1e-12)
        bad += level_score(arch, D, x, lv, V)
    return bad + check_rho()


def level_score(arch: str, D, x, lv, V) -> list:
    bad, x0 = [], RL.x0_of(V, arch)
    for a in months(x):
        P = RL.pool_of(arch, D, x, a)
        c, m = RL.clip_stats(x0, P)
        s, v = torch.zeros((), dtype=torch.float64, requires_grad=True), torch.zeros(RL.n_par(arch), dtype=torch.float64, requires_grad=True)
        RL.loss_r(v, RL.view(D, lv, P), arch, 0.0, c, m, "R", True, shift=s).backward()
        par = float(v.grad[0]) if arch == "A" else float(v.grad[2] + v.grad[3])
        print("  %s %s: |P| = %d, c = %.3f, m = %.4f; балл сдвига ln F = %.2e; параметр уровня (δ₀ / θ_c0 + θ_p0, с весами π·(1 − w) или π·ŝ) = %.2e" % (arch, pd.DatetimeIndex(x["d"])[a].date(), len(P), c, m, float(s.grad), par))
        bad += ["балл уровня %s %.2e" % (arch, float(s.grad))] * (abs(float(s.grad)) > 1e-12)
    return bad


def check_grad() -> list:
    bad = []
    for arch in ("A", "CK"):
        D, x, lv = load(arch)
        x0 = RL.x0_of(RL.view(D, lv, np.arange(len(lv.pi))), arch)
        P = RL.pool_of(arch, D, x, months(x)[1])
        V, (c, m) = RL.view(D, lv, P), RL.clip_stats(x0, P)
        starts, _ = RL._starts(V, arch, (1, 2, 3, 4, 5), True)
        n_hi = []
        for x_ in starts:
            f = lambda v: RL.loss_r(v, V, arch, 1e-2, c, m, "R")
            v = torch.as_tensor(x_).clone().requires_grad_(True)
            f(v).backward()
            fd = np.array([(float(f(torch.as_tensor(x_ + h))) - float(f(torch.as_tensor(x_ - h)))) / 2e-6 for h in 1e-6 * np.eye(len(x_))])
            g = v.grad.numpy()
            err = float(np.max(np.abs(g - fd)) / max(np.max(np.abs(g)), 1e-8))
            u = (torch.log(V.rv) - RL.ln_f(RL.lnc_of(torch.as_tensor(x_), V, arch), V) - math.log(m)).numpy()
            n_hi.append(int((u > math.log(c / m)).sum()))
            bad += ["градиент %s: %.2e" % (arch, err)] * (err > 1e-6)
            print("  %s: отн. расхождение autograd / разностей = %.2e; пар с u > ln κ: %d" % (arch, err, n_hi[-1]))
        bad += ["%s: < 2 точек с u > ln κ" % arch] * (sum(k > 0 for k in n_hi) < 2)
    return bad


def corrupt(D, arch: str, lv, mask) -> tuple:
    Dc, name = copy.copy(D), "rv_next" if arch == "A" else "rv"
    a = np.array(getattr(D, name), float)
    a[mask] *= 7.0
    setattr(Dc, name, a)
    return Dc, RL.Lv(np.where(mask, 0.3, lv.pi), lv.lnb, np.where(mask, lv.bg + 0.5, lv.bg), lv.has)


def check_pit() -> list:
    bad = []
    for arch in ("A", "CK"):
        D, x, lv = load(arch)
        n, v = len(lv.pi), torch.as_tensor(RL._starts(RL.view(D, lv, [0]), arch, (2,), True)[0][0])
        x0 = RL.x0_of(RL.view(D, lv, np.arange(n)), arch)
        for a in months(x):
            P = RL.pool_of(arch, D, x, a)
            for tag, T, bound in (("месяц D", P, P), ("блок V_3", P[:-126], P[:-126])):
                mask = np.ones(n, bool)
                mask[bound] = False
                Dc, lc = corrupt(D, arch, lv, mask)
                x1 = RL.x0_of(RL.view(Dc, lc, np.arange(n)), arch)
                s0, s1 = RL.clip_stats(x0, T), RL.clip_stats(x1, T)
                l0 = float(RL.loss_r(v, RL.view(D, lv, T), arch, 1e-2, *s0, "R"))
                l1 = float(RL.loss_r(v, RL.view(Dc, lc, T), arch, 1e-2, *s1, "R"))
                ok = s0 == s1 and l0 == l1
                print("  %s %s %s: c, m, L_R %s" % (arch, tag, pd.DatetimeIndex(x["d"])[a].date(), "не изменились" if ok else "ИЗМЕНИЛИСЬ"))
                bad += ["PIT %s %s" % (arch, tag)] * (not ok)
    return bad


def main(args: list) -> int:
    from models_v2.s18.r_loss_syn import synth
    fn = {"--check-nest": check_nest, "--check-grad": check_grad, "--check-pit": check_pit, "--synth": synth}
    k = [a for a in args if a in fn]
    if not k:
        raise SystemExit("укажите --check-nest, --check-grad, --check-pit или --synth")
    bad = fn[k[0]]()
    print("%s: %s" % (k[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
