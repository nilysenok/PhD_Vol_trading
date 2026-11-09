"""V6-1a, G14: охват и стресс-окна (план V6, п. 2.3). Охват: вес w ∈ [0, 1] в G = B^(1−w)·F^w, минимизирующий mean QLIKE (производная ℓ·(1 − rv/B·exp(−wℓ)), ℓ = ln(F/B), монотонна — деление пополам);
интервал 90 % — блоковый бутстрап по датам (блок 20, 2000, seed 0), веса как счётчики дат. Стресс-окна: без 2017, без 2017–2018, без 1 % крайних дат, кризисные (2020-02-20…2020-06-30, 2022-02-21…2022-06-30 по t⁺),
дни с H в верхних 10 % своего года (год t⁺), спокойные — H в нижних 50 % своего года; везде — Δ в % уровня K1 (в окне)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV

CRISIS = (("2020-02-20", "2020-06-30"), ("2022-02-21", "2022-06-30"))


def solve_w(rv_e, ell, cnt, iters: int = 60) -> np.ndarray:
    """Вес охвата для каждой строки cnt (число повторов дат) — деление пополам на [0, 1] по знаку производной."""
    lo, hi = np.zeros(len(cnt)), np.ones(len(cnt))
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        grad = (cnt * (ell * (1.0 - rv_e * np.exp(-m[:, None] * ell)))).sum(1)
        lo, hi = np.where(grad < 0, m, lo), np.where(grad < 0, hi, m)
    return 0.5 * (lo + hi)


def coverage(rv, base, f, b: int = EV.B, block: int = EV.BLOCK, seed: int = EV.SEED) -> dict:
    """Вес w в B^(1−w)·F^w и 90 % интервал; w = 0 — F ничего не добавляет к B, w = 1 — B ничего не добавляет к F."""
    rv, base, f = (np.asarray(x, float) for x in (rv, base, f))
    ell, rv_e = np.log(f / base), rv / base
    n = len(rv)
    w0 = float(solve_w(rv_e, ell, np.ones((1, n)))[0])
    cnt = np.stack([np.bincount(i, minlength=n) for i in SD.block_idx(n, B_=b, block=block, seed=seed)]).astype(float)
    lo, hi = np.quantile(solve_w(rv_e, ell, cnt), [0.05, 0.95])
    return dict(w=w0, lo90=float(lo), hi90=float(hi))


def stress_masks(tp, h, q_f, q_best) -> dict:
    """Маски сценариев на общих датах. h — H дня d; квантили H — внутри года цели t⁺; «крайние» — 1 % дат окна 2017–2021 с наибольшей |q_f − q_best| (q_best — лучший из K2/K3)."""
    tp = pd.DatetimeIndex(tp)
    y, h = tp.year.to_numpy(), np.asarray(h, float)
    w = Q.wmask(tp, "2017_2021")
    out = {"без 2017 (2018–2021)": (y >= 2018) & (y <= 2021), "без 2017–2018 (2019–2021)": (y >= 2019) & (y <= 2021)}
    ext = np.zeros(len(tp), bool)
    idx = np.flatnonzero(w)
    drop = idx[np.argsort(-np.abs(np.asarray(q_f)[idx] - np.asarray(q_best)[idx]), kind="stable")[: int(round(Q.TRIM * len(idx)))]]
    ext[drop] = True
    out["без 1 % крайних дат (2017–2021)"] = w & ~ext
    for a, b2 in CRISIS:
        out["кризис %s … %s" % (a, b2)] = (tp >= a) & (tp <= b2)
    hi, calm = np.zeros(len(tp), bool), np.zeros(len(tp), bool)
    for k in np.unique(y):
        s = y == k
        hi[s] = h[s] >= np.quantile(h[s], 0.9)
        calm[s] = h[s] <= np.quantile(h[s], 0.5)
    out["H в верхних 10 % года"], out["спокойные: H в нижних 50 % года"] = hi, calm
    return out


def stress_table(rv, f, refs: dict, tp, h) -> pd.DataFrame:
    """Строка на сценарий: n, QLIKE ряда, Δ % уровня K1 к каждому ref."""
    rv, f = np.asarray(rv, float), np.asarray(f, float)
    qf = Q.ql(rv, f)
    qb = np.minimum(Q.ql(rv, refs["K2"]), Q.ql(rv, refs["K3"]))
    rows = []
    for name, m in stress_masks(tp, h, qf, qb).items():
        k1 = float(Q.ql(rv[m], np.asarray(refs["K1"])[m]).mean())
        r = dict(scenario=name, n=int(m.sum()), qlike=float(qf[m].mean()), qlike_K1=k1)
        r.update({"d_pct_%s" % k: 100 * float((qf[m] - Q.ql(rv[m], np.asarray(v)[m])).mean()) / k1 for k, v in refs.items()})
        rows.append(r)
    return pd.DataFrame(rows)
