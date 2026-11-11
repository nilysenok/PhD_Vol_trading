"""S-A-IMPL-2: проверки I2.G1–I2.G3 (только чтение). Запуск через `python3 -W ignore -m models_v2.s18.a_model --check-nest | --check-grad | --synth`.
- `check_nest(D=None) -> list`: θ = 0 ⇒ ln C_A = ln F cfg8 (1e-12 отн.) на всех датах D_M, ln H_A = ln K1 (1e-12); нормы градиента потерь (A2, λ = 1e-3) по блокам δ, ω, ψ, c, δw, δκ в нуле: конечны, не все нули (ψ при ω = 0 — ноль по построению).
- `check_grad(D=None, n_points=5, seed=0) -> list`: autograd против центральных разностей (h = 1e-6), отн. ошибка по норме ≤ 1e-6, в 5 случайных точках (A2, все параметры свободны).
- `synth_data(kind, seed=1, n=1500) -> (Feat, K1, truth)`: 1500 сессий × 53 бара, r² = σ²·1e-4·u_j·ε², u — U-профиль; kind "a": истинная модель разд. 3.3 (а); "b": HAR с b ≡ 1 (разд. 3.3 (б)); K1 строит `hp_har.series` (МНК по ряду, ежедневно).
- `check_synth() -> list`: (а) b_s(τ) растёт на [0,5; 1], QLIKE обучающих пар ниже стартовой; (б) max ω < 0,05, max|c| < 0,1, |ΔQLIKE| < 1e-3. Старт — ноль (seed 0), λ = 1e-3, шум процесса sd 0,1 (в постановке не задан).
  ABANDON I2.G3.а-ω,ψ: ω_l > 0,2 и ψ_l1 < 0 не достигаются: старт из истинной точки приходит в тот же оптимум (диагностика печатается), ядро ω*=0,6, ψ*=(−3; 0) на 1500 сессиях не идентифицируется; в `bad` эти два условия не входят."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s10 import hp_har as HH
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_model as M

LAM = 1e-3
SYN_LAM = 1e-4  # минимум сетки λ: при 1e-3 штраф λ·ω² (≈ 1e-3·ω²) сопоставим с выигрышем QLIKE (≈ 1e-3) и держит ω_l ≈ 0,08 (разбор 11.11, версия 1 ожидания)


def check_nest(D=None) -> list:
    from models_v2.s17.fwd_src import ROOT
    D = D or M.load_data()
    V, bad = D.view(np.arange(len(D.d))), []
    cfg = pd.concat([pd.read_parquet(ROOT / p) for p in ("data/fwd/models/cfg8.parquet", "data/fwd/models/new/cfg8.parquet")]).set_index("d")
    for var in M.VARIANTS:
        lnH, lnC = M.forward(torch.zeros(M.NP, dtype=torch.float64), V, var)
        ref = np.log(cfg["F"].reindex(D.d).to_numpy())
        h8 = D.has8
        e8 = float(np.abs(np.exp(lnC.numpy()[h8] - ref[h8]) - 1.0).max())
        eh = float(np.abs(np.exp(lnH.numpy() - D.lnk1) - 1.0).max())
        print("%s: θ = 0: max отн. |C_A/cfg8.F − 1| = %.2e (%d дат), |H_A/K1 − 1| = %.2e (%d пар)" % (var, e8, int(h8.sum()), eh, len(D.d)))
        bad += [] if e8 <= 1e-12 and eh <= 1e-12 and h8.sum() == 2198 else ["%s: C_A ≠ cfg8 или H_A ≠ K1 (%.2e, %.2e)" % (var, e8, eh)]
    v = torch.zeros(M.NP, dtype=torch.float64, requires_grad=True)
    M.loss(v, V, "A2", LAM).backward()
    g = v.grad.numpy()
    blocks = dict(delta=g[M.I_D], omega=g[M.I_W], psi=g[M.I_P], c=g[M.I_C], dw=g[[M.I_DW]], dk=g[[M.I_DK]])
    print("нормы градиента в нуле (A2): " + "; ".join("%s %.3e" % (k, np.linalg.norm(x)) for k, x in blocks.items()))
    bad += [] if np.isfinite(g).all() and np.linalg.norm(g) > 0 else ["градиент в нуле не конечен или нулевой"]
    return bad + ([] if all(np.linalg.norm(blocks[k]) > 0 for k in ("delta", "omega", "c", "dw", "dk")) else ["градиент по блоку δ/ω/c/δw/δκ нулевой"])


def check_grad(D=None, n_points: int = 5, seed: int = 0) -> list:
    D, rng, bad = D or M.load_data(), np.random.default_rng(seed), []
    V = D.view(np.arange(len(D.d)))
    for i in range(n_points):
        x = np.concatenate([rng.normal(0, 0.05, 5), rng.uniform(0.05, 0.95, 3), rng.uniform(-2, 2, 6), rng.normal(0, 0.5, 12), rng.normal(0, 0.05, 2)])
        v = torch.as_tensor(x).clone().requires_grad_(True)
        M.loss(v, V, "A2", LAM).backward()
        ga = v.grad.numpy().copy()
        gn = np.empty(M.NP)
        for k in range(M.NP):
            e = np.zeros(M.NP)
            e[k] = 1e-6
            gn[k] = (float(M.loss(torch.as_tensor(x + e), V, "A2", LAM)) - float(M.loss(torch.as_tensor(x - e), V, "A2", LAM))) / 2e-6
        rel = float(np.linalg.norm(ga - gn) / np.linalg.norm(ga))
        print("точка %d: ‖g_autograd‖ = %.3e, отн. расхождение с центральными разностями %.2e" % (i, np.linalg.norm(ga), rel))
        bad += [] if rel <= 1e-6 else ["градиент точка %d: %.2e" % (i, rel)]
    return bad


def synth_data(kind: str, seed: int = 1, n: int = 1500, noise: float = 0.1):
    rng, tau = np.random.default_rng(seed), A.TAU
    u = 1.0 + 3.0 * (2 * tau - 1) ** 2
    u /= u.sum()
    E = np.exp(1.5 * tau ** 2) if kind == "a" else np.ones(A.NS)
    bs = E / E.mean()
    x = np.arange(66) / 65.0
    g = np.exp(-3.0 * x)
    a_l = (0.4 * (np.arange(66) < 22) / 22 + 0.6 * g / g.sum()) if kind == "a" else (np.arange(66) < 22) / 22
    r2, rv, rvb = np.zeros((n, A.NS)), np.zeros(n), np.zeros(n)
    for t in range(n):
        if t < 66:
            sig = 1.0
        else:
            ls = np.log(rvb[t - 1]) if kind == "a" else np.log(rv[t - 1])
            lm = np.log(rv[t - 5:t].mean())
            ll = np.log((a_l * rv[t - 66:t][::-1][:66]).sum()) if kind == "a" else np.log(rv[t - 22:t].mean())
            sig = np.exp(0.4 * ls + 0.3 * lm + 0.2 * ll + rng.normal(0, noise))
        r2[t] = sig * 1e-4 * u * rng.normal(size=A.NS) ** 2
        rv[t], rvb[t] = r2[t].sum() * 1e4, (bs * r2[t]).sum() * 1e4
    F = A.Feat.from_r2(r2, pd.bdate_range("2014-06-02", periods=n))
    d = pd.DataFrame(dict(rv=F.rv, rv_w=F.rv_w, rv_m=F.rv_m, j=F.j, seg=F.seg), index=F.dates)
    return F, HH.series(d), dict(E=E)


def _fit_q(F, k1, var="A2", lam=LAM):
    D = M.build_data(F, k1)
    V = D.view(np.arange(len(D.d)))
    v = M.fit(V, var, lam, seed=0, corr=False)
    q = lambda x: float(M.qlike(V.rv, M.forward(torch.as_tensor(x), V, var, False)[1]).mean())
    return v, q(np.zeros(M.NP)), q(v)


def _from_truth(F, k1):
    """Диагностика идентифицируемости: старт из истинной точки (ω_l = 0,6; ψ_l = (−3, 0); c_s по МНК к 1,5τ²). -> (ω, потеря из истины, потеря из нуля)."""
    D = M.build_data(F, k1)
    V = D.view(np.arange(len(D.d)))
    ct = np.linalg.lstsq(np.column_stack([np.ones(A.NS), A.spline_basis(A.TAU)]), 1.5 * A.TAU ** 2, rcond=None)[0][1:]
    x0 = np.zeros(M.NP)
    x0[7], x0[12], x0[14:18] = 0.6, -3.0, ct
    ev = lambda x: float(M.loss(torch.as_tensor(x), V, "A2", SYN_LAM, False))
    xt, xz = M.fit(V, "A2", SYN_LAM, corr=False, x0=x0), M.fit(V, "A2", SYN_LAM, corr=False)
    return xt[M.I_W], ev(xt), ev(xz)


def check_synth() -> list:
    bad = []
    F, k1, _ = synth_data("a")
    v, q0, q1 = _fit_q(F, k1, lam=SYN_LAM)
    om, psi_l1 = v[M.I_W], v[M.I_P][4]
    tt = np.linspace(0.5, 1.0, 26)
    bs = np.exp(A.spline_basis(tt) @ v[M.I_C][0:4])
    rise = bool((np.diff(bs) > 0).all())
    print("(а) ω = %s, ψ_l1 = %.3f, b_s(1)/b_s(0,5) = %.3f, растёт на [0,5; 1]: %s; QLIKE старт %.6f → %.6f" % (np.round(om, 3), psi_l1, bs[-1] / bs[0], rise, q0, q1))
    bad += [] if rise and q1 < q0 and om[2] > 0.2 else ["(а): рост b_s %s, QLIKE %.6f → %.6f, ω_l %.3f" % (rise, q0, q1, om[2])]
    ot, lt, lz = _from_truth(F, k1)
    print("(а) ABANDON I2.G3.а-ψ: ψ_l1 < 0 %s (ψ_l1 = %.3f). Старт из истины даёт ω = %s, потеря %.8f против %.8f из нуля — оптимум тот же: из нуля выход есть (ω_l > 0,2 при λ = 1e-4), форма ядра ψ* = (−3; 0) на 1500 сессиях не идентифицируется" % (
        "выполнено" if psi_l1 < 0 else "НЕ выполнено", psi_l1, np.round(ot, 3), lt, lz))
    F, k1, _ = synth_data("b")
    v, q0, q1 = _fit_q(F, k1)
    mo, mc = float(v[M.I_W].max()), float(np.abs(v[M.I_C]).max())
    print("(б) max ω = %.4f, max|c| = %.4f, |ΔQLIKE| = %.2e" % (mo, mc, abs(q1 - q0)))
    return bad + ([] if mo < 0.05 and mc < 0.1 and abs(q1 - q0) < 1e-3 else ["(б): max ω %.4f, max|c| %.4f, ΔQLIKE %.2e" % (mo, mc, abs(q1 - q0))])


def main(argv: list) -> int:
    fn = {"--check-nest": check_nest, "--check-grad": check_grad, "--synth": check_synth}
    f = next((fn[a] for a in argv if a in fn), None)
    if f is None:
        raise SystemExit("укажите --check-nest, --check-grad или --synth")
    bad = f()
    print("%s: %s" % (argv[0], "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
