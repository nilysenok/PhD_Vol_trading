"""S-A-IMPL-2 (Ж-254; `S_A_SPEC.md`, разд. 2–4): H_A, C_A, функция потерь, штраф, L-BFGS-B. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Только чтение K1, cfg8, x1.
Вектор параметров θ (28, float64): δ = θ[0:5] = (δ₀, δ_s, δ_m, δ_l, δ_J); ω = θ[5:8]; ψ = θ[8:14] (по k: ψ₁, ψ₂); c = θ[14:26] (по k: 4 коэффициента сплайна); δw = θ[26]; δκ = θ[27]. Ноль = контроль (a = HAR, b ≡ 1, корректор cfg8).
Публичное API:
- `VARIANTS = ("A0", "A1", "A2", "A2p")`; `free_mask(var, corr) -> bool[28]` (A0: δ, [δw, δκ]; A1: + ω, ψ; A2, A2p: + c; δw, δκ — только при `corr`); `tau_of(var)` — TAU или TAU[PERM] (A2p).
- `Data` — пары K1 без cross_seg (2451 строка: d, t⁺ = target_date, rv_next = rv целевого дня, lnK1, β^{K1} = (b_d, b_w, b_m), ln(1+J_d), lnS0 = ln(rv, rv_w, rv_m) строки d, lag-индексы), cfg8-колонки (w, m = G − w·ℓ, ln B_mix, has8 — d ≥ 2018-01-03).
  `build_data(F, k1, cfg=None, bmix=None) -> Data`; `load_inputs() -> (K1, cfg8, B_mix)`; `load_data(F=None) -> Data` — боевой вход (K1, cfg8 архив + new, B_mix = колонка M x1). `Data.view(sel) -> View` (sel — позиции пар).
- `forward(v, view, var, corr=True) -> (lnH, lnC)` (torch): ln H_A = ln K1 + δ₀ + Σ(β_k + δ_k)·ln S_k(a, b) − Σβ_k·ln S_k(HAR, 1) + δ_J·ln(1+J); ℓ_A = ln B_mix − ln H_A;
  ln C_A = ln H_A + w_eff·ℓ_A + (1 + δκ)·m (только пары с cfg8; иначе lnC = lnH: g = 0), w_eff = w + δw, зажатый в [min(0, w), max(1, w)] (сохранённый w cfg8 бывает −0,02 на 42 днях — при δw = 0 вложение точное).
  A0 использует ln S(HAR, 1) = ln(rv, rv_w, rv_m) без a, b.
- `penalty(v, lam) -> torch`: λ(‖δ‖² + Σω² + Σ‖c‖² + δw² + δκ²) + 1e-4·Σ‖ψ‖². `loss(v, view, var, lam, corr) -> torch`: среднее QLIKE (по lnC для пар с cfg8, иначе по lnH) + penalty.
- `fit(view, var, lam, seed=0, corr=True, maxiter=500, x0=None) -> np.ndarray(28)` (x0 — диагностический старт вместо нуля/seed): scipy L-BFGS-B (границы: ω ∈ [0,1], ψ ∈ [−8,8], δw ∈ [−max(min w, 0), 1 − min(max w, 1)] по обучающим парам, δκ ∈ [−1,1]; несвободные = 0, ftol 1e-10), градиент torch autograd.
  seed 0 — точный ноль; seeds 1–4 — ноль плюс N(0; 0,05²) на ω, ψ, c (вектор из 21 числа, одинаковый для всех вариантов; ω затем clip [0,1], несвободные обнулены).
- `qlike(rv, lnF) -> torch`. Синтетика и проверки — `a_model_chk.py`.
Запуск: `python3 -W ignore -m models_v2.s18.a_model --check-nest | --check-grad | --synth` (только чтение)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from models_v2.s18 import a_feat as A

VARIANTS = ("A0", "A1", "A2", "A2p")
NP, SEED_SD, PSI_PEN = 28, 0.05, 1e-4
I_D, I_W, I_P, I_C, I_DW, I_DK = slice(0, 5), slice(5, 8), slice(8, 14), slice(14, 26), 26, 27
CFG8_FIRST = pd.Timestamp("2018-01-03")


def free_mask(var: str, corr: bool) -> np.ndarray:
    m = np.zeros(NP, bool)
    m[I_D] = True
    m[I_W] = m[I_P] = var != "A0"
    m[I_C] = var in ("A2", "A2p")
    m[I_DW] = m[I_DK] = corr
    return m


def tau_of(var: str) -> np.ndarray:
    return A.slot_tau(var == "A2p")


class View:
    def __init__(self, D: "Data", sel):
        sel = np.asarray(sel)
        t = lambda a: torch.as_tensor(np.asarray(a, float)[sel])
        self.F, self.sel, self.tau_var = D.F, sel, None
        self.rv, self.lnk1, self.beta, self.lnj, self.lnS0 = t(D.rv_next), t(D.lnk1), t(D.beta), t(D.ln1pj), t(D.lnS0)
        self.w, self.m, self.lnb, self.has8 = t(D.w), t(D.m), t(D.lnb), torch.as_tensor(D.has8[sel])
        self.lag = [(ix[sel], ok[sel]) for ix, ok in D.lag]


class Data:
    pass


def build_data(F, k1: pd.DataFrame, cfg: pd.DataFrame | None = None, bmix: pd.Series | None = None) -> Data:
    k = k1[~k1["cross_seg"]].reset_index(drop=True)
    D = Data()
    D.F, D.d, D.tp = F, k["d"].to_numpy(), k["target_date"].to_numpy()
    D.rows = F.dates.get_indexer(k["d"])
    if (D.rows < 0).any():
        raise SystemExit("дата K1 не в дневном ряду")
    r = D.rows
    D.rv_next, D.lnk1 = k["rv"].to_numpy(float), np.log(k["sigma2"].to_numpy(float))
    D.beta, D.ln1pj = k[["b_d", "b_w", "b_m"]].to_numpy(float), np.log1p(F.j[r])
    D.lnS0 = np.log(np.column_stack([F.rv[r], F.rv_w[r], F.rv_m[r]]))
    D.lag = A.lag_idx(F, r)
    n = len(k)
    D.w, D.m, D.lnb, D.has8 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, bool)
    if cfg is not None:
        c = cfg.set_index("d")
        got = k["d"].isin(c.index).to_numpy()
        w, G, ell = (c.reindex(k["d"])[x].to_numpy(float) for x in ("w", "G", "ell"))
        D.has8, D.w = got, np.where(got, w, 0.0)
        D.m = np.where(got, G - w * ell, 0.0)
        D.lnb = np.where(got, np.log(bmix.reindex(k["d"]).to_numpy(float)), 0.0)
    return D


Data.view = lambda self, sel: View(self, sel)


def load_inputs() -> tuple:
    """(K1, cfg8 архив + new, B_mix = колонка M x1 по d) — сохранённые PIT-ряды, только чтение."""
    from models_v2.s16 import x1_w3_core as C
    from models_v2.s17.fwd_src import ROOT
    cfg = pd.concat([pd.read_parquet(ROOT / p) for p in ("data/fwd/models/cfg8.parquet", "data/fwd/models/new/cfg8.parquet")], ignore_index=True)
    return A.load_k1(), cfg, C.load().set_index("d")["M"]


def load_data(F=None) -> Data:
    return build_data(F or A.build_sber(), *load_inputs())


def unpack(v) -> dict:
    return dict(omega=v[I_W], psi=v[I_P].reshape(3, 2), c=v[I_C].reshape(3, 4))


def forward(v, V: View, var: str, corr: bool = True):
    dl = v[I_D]
    if var == "A0":
        lnS = V.lnS0
    else:
        lnS = A.ln_S(V.F, unpack(v), V.lag, tau_of(var))
    lnH = V.lnk1 + dl[0] + ((V.beta + dl[1:4]) * lnS).sum(1) - (V.beta * V.lnS0).sum(1) + dl[4] * V.lnj
    w = torch.minimum(torch.maximum(V.w + v[I_DW], torch.clamp(V.w, max=0.0)), torch.clamp(V.w, min=1.0))
    lnC = lnH + w * (V.lnb - lnH) + (1.0 + v[I_DK]) * V.m
    return lnH, torch.where(V.has8, lnC, lnH)


def qlike(rv, lnF):
    return rv * torch.exp(-lnF) + lnF - torch.log(rv) - 1.0


def penalty(v, lam):
    return lam * ((v[I_D] ** 2).sum() + (v[I_W] ** 2).sum() + (v[I_C] ** 2).sum() + v[I_DW] ** 2 + v[I_DK] ** 2) + PSI_PEN * (v[I_P] ** 2).sum()


def loss(v, V: View, var: str, lam: float, corr: bool = True):
    return qlike(V.rv, forward(v, V, var, corr)[1]).mean() + penalty(v, lam)


def _x0(var: str, seed: int, free: np.ndarray) -> np.ndarray:
    x = np.zeros(NP)
    if seed:
        z = np.random.default_rng(seed).normal(0.0, SEED_SD, 21)
        x[I_W], x[I_P], x[I_C] = z[:3], z[3:9], z[9:21]
        x[I_W] = np.clip(x[I_W], 0.0, 1.0)
    return np.where(free, x, 0.0)


def _bounds(V: View, free: np.ndarray) -> list:
    b = [(None, None)] * NP
    b[5:8], b[8:14], b[I_DK] = [(0.0, 1.0)] * 3, [(-8.0, 8.0)] * 6, (-1.0, 1.0)
    if V.has8.any():
        w = V.w[V.has8]
        b[I_DW] = (-max(float(w.min()), 0.0), 1.0 - min(float(w.max()), 1.0))
    return [bb if f else (0.0, 0.0) for bb, f in zip(b, free)]


def fit(V: View, var: str, lam: float, seed: int = 0, corr: bool = True, maxiter: int = 500, x0=None) -> np.ndarray:
    free = free_mask(var, corr)

    def fun(x):
        v = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss(v, V, var, lam, corr)
        f.backward()
        return float(f), v.grad.numpy().astype(float)
    r = minimize(fun, _x0(var, seed, free) if x0 is None else np.where(free, x0, 0.0), jac=True, method="L-BFGS-B", bounds=_bounds(V, free), options=dict(maxiter=maxiter, ftol=1e-10))
    return r.x


if __name__ == "__main__":
    from models_v2.s18 import a_model_chk
    sys.exit(a_model_chk.main(sys.argv[1:]))
