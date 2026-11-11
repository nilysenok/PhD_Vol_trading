"""Проверки S-C-IMPL-2 (`cm_model`): --check-nest (M.G1), --check-grad (M.G2), --synth (M.G3). Только чтение; синтетика — детерминирована (rng по seed).
Синтетика (выбор исполнителя, в спецификации не задано): RV = C·χ²₁₀/10 (ν = 10), ln C₈ ~ N(0; 0,5²), признаки ln(Ĥ/A) ~ N(0; 1), ln(X_d/X_m) ~ N(0; 0,5²); (а) λ = 1e-4, (б) λ = 1e-2, (в) λ = 1e-4."""
from __future__ import annotations

import numpy as np
import torch

from models_v2.s18 import cm_model as M

REL = 1e-13
THETA_STAR = np.array([0.4, 0.6, 0.1, -0.1, -0.3, 0.2])       # γ_c, γ_p, θ_c0, θ_p0, θ_c1, θ_p1


def real_data():
    from models_v2.s16 import x1_w3_core as C
    from models_v2.s18 import cm_feat as CF
    x, feat = C.load(), CF.load_feat()
    return {v: M.build_data(x, feat, lag=v == "CKL") for v in M.TRAINED}


def _rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1.0)))


def check_nest() -> list:
    bad, Ds = [], real_data()
    z = torch.zeros(M.NP, dtype=torch.float64)
    for var, D in Ds.items():
        V = D.view(np.arange(D.n))
        f = M.forward(z, V, var)
        same = np.array_equal(D.c8 * f["M"].numpy(), D.c8)
        print("%s: n = %d, θ = 0 ⇒ C_V = cfg8 побитно: %s" % (var, D.n, same))
        bad += [] if same else ["%s: θ = 0 не даёт cfg8 побитно" % var]
        v0 = z.clone().requires_grad_(True)
        M.loss(v0, V, var, 1e-3).backward()
        g = v0.grad.numpy()
        print("%s: ‖∇L(0)‖ = %.3e, ненулевых %d из 6" % (var, np.linalg.norm(g), int((g != 0).sum())))
        bad += [] if np.isfinite(g).all() and (g != 0).any() else ["%s: градиент в нуле нефинитен или нулевой" % var]
    V = Ds["CK"].view(np.arange(Ds["CK"].n))
    c8, c7 = V.c8.numpy(), M.forward(z, V, "C7")["M"].numpy() * V.c8.numpy()
    ck = c8 * M.forward(torch.tensor([1.0, 1.0, 0, 0, 0, 0], dtype=torch.float64), V, "CK")["M"].numpy()
    print("γ = 1, θ = 0: C_K vs C_7 max отн. %.2e" % _rel(ck, c7))
    bad += [] if _rel(ck, c7) <= REL else ["C_K(γ = 1) ≠ C_7"]
    rng = np.random.default_rng(1)
    for var, D in Ds.items():
        if var == "CD":
            continue
        v = torch.as_tensor(np.clip(rng.normal(0, 0.4, M.NP), [0, 0, -3, -3, -3, -3], 1))
        f = M.forward(v, D.view(np.arange(D.n)), var)
        e = _rel(f["Fc"].numpy() + f["Fp"].numpy() + f["Fu"].numpy(), f["M"].numpy() * D.c8)
        print("%s: ΣF_X + F_u = C: max отн. %.2e" % (var, e))
        bad += [] if e <= REL else ["%s: ΣF ≠ C" % var]
    return bad


def check_grad() -> list:
    bad, rng, h = [], np.random.default_rng(2), 1e-6
    for var, D in real_data().items():
        V = D.view(np.arange(D.n))
        mu = M.MU[var]
        for k in range(5):
            x = np.array([rng.uniform(lo + 0.1, min(hi, lo + 0.9)) if hi - lo == 1 else rng.normal(0, 0.4) for lo, hi in M.bounds(var)])
            v = torch.as_tensor(x).clone().requires_grad_(True)
            M.loss(v, V, var, 1e-3, mu).backward()
            g = v.grad.numpy()
            fd = np.array([(float(M.loss(torch.as_tensor(x + h * e), V, var, 1e-3, mu)) - float(M.loss(torch.as_tensor(x - h * e), V, var, 1e-3, mu))) / (2 * h) for e in np.eye(M.NP)])
            r = float(np.abs(g - fd).max() / np.linalg.norm(g))
            bad += [] if r <= 1e-6 else ["%s: точка %d, отн. %.2e" % (var, k, r)]
            print("%s точка %d: autograd vs центр. разности max отн. %.2e" % (var, k, r))
    return bad


def synth(n: int, theta, seed: int, su: float = 0.0, null: bool = False, nu: int = 10) -> M.Data:
    g = np.random.default_rng(seed)
    c8 = np.exp(g.normal(0, 0.5, n))
    sc = np.clip(g.beta(3, 4, n) * (1 - su), 0.05, 0.95 * (1 - su)) if null else g.uniform(0.2, 0.7 if su == 0 else 0.4, n)
    sp = 1 - su - sc
    xc, xp, zc, zp = g.normal(0, 1, n), g.normal(0, 1, n), g.normal(0, 0.5, n), g.normal(0, 0.5, n)
    t = np.asarray(theta, float)
    ec, ep = t[0] * xc + t[2] + t[4] * zc, t[1] * xp + t[3] + t[5] * zp
    Mm = 1 + sc * np.expm1(ec) + sp * np.expm1(ep)
    ch = lambda: g.chisquare(nu, n) / nu
    return M.Data(rv=c8 * Mm * ch(), c8=c8, Hc=sc * c8 * np.exp(xc), Hp=sp * c8 * np.exp(xp), sc=sc, sp=sp, su=np.full(n, su), zc=zc, zp=zp,
                  labc=sc * c8 * np.exp(ec) * ch(), labp=sp * c8 * np.exp(ep) * ch())


def synth_check() -> list:
    bad, z = [], torch.zeros(M.NP, dtype=torch.float64)
    for var in ("CK", "CKA"):
        D = synth(2500, THETA_STAR, 11)
        V = D.view(np.arange(D.n))
        th = M.fit(V, var, 1e-4)["theta"]
        e, q0, q1 = np.abs(th - THETA_STAR).max(), float(M.qlike_sum(z, V, var)), float(M.qlike_sum(torch.as_tensor(th), V, var))
        print("(а) %s: max|θ̂ − θ*| = %.3f; QLIKE %.5f → %.5f" % (var, e, q0, q1))
        bad += [] if e <= 0.1 and q1 < q0 else ["(а) %s: восстановление / QLIKE" % var]
    D = synth(2500, np.zeros(M.NP), 12, null=True)
    V = D.view(np.arange(D.n))
    for var in ("CK", "CKA", "CD"):
        th = M.fit(V, var, 1e-2)["theta"]
        dq = float(M.qlike_sum(z, V, var)) - float(M.qlike_sum(torch.as_tensor(th), V, var))
        print("(б) %s: max|θ̂| = %.4f; |ΔQLIKE| = %.2e" % (var, np.abs(th).max(), abs(dq)))
        bad += [] if np.abs(th).max() < 0.05 and abs(dq) < 1e-3 else ["(б) %s: нулевой сигнал не нулевой" % var]
    D = synth(2500, THETA_STAR, 13, su=0.3)
    V = D.view(np.arange(D.n))
    f = M.forward(torch.as_tensor(THETA_STAR), V, "CK")
    s = float(np.max(np.abs((f["Fc"] + f["Fp"] + f["Fu"]).numpy() / (f["M"] * V.c8).numpy() - 1)))
    th = M.fit(V, "CK", 1e-4)["theta"]
    q_ok, q_mut = float(M.qlike(V.rv, f["Fc"] + f["Fp"] + f["Fu"]).mean()), float(M.qlike(V.rv, f["Fc"] + f["Fp"]).mean())
    print("(в) ŝ_u = 0,3: ΣF = C отн. %.2e; max|θ̂ − θ*| = %.3f; QLIKE(θ*) честная %.5f, без u %.5f" % (s, np.abs(th - THETA_STAR).max(), q_ok, q_mut))
    bad += [] if s <= REL and np.abs(th - THETA_STAR).max() <= 0.1 and q_ok < q_mut else ["(в) uncovered"]
    return bad


def main(argv) -> int:
    key = next((k for k in ("--check-nest", "--check-grad", "--synth") if k in argv), None)
    if key is None:
        raise SystemExit("укажите --check-nest, --check-grad или --synth")
    bad = {"--check-nest": check_nest, "--check-grad": check_grad, "--synth": synth_check}[key]()
    print("%s: %s" % (key, "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
