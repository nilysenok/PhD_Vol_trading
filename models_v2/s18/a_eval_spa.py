"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 6.3, 8): студентизированный SPA (Hansen 2005) и StepM (Romano–Wolf 2005) — по спецификации, на дневных потерях; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Бенчмарк l_bench, кандидаты L (n, K); d_k = l_bench − l_k, плюс = кандидат лучше; H0: max_k E d_k ≤ 0. Бутстрап — стационарный (`a_eval_stat.stat_idx`, блок 20, B = 10 000, seed 0).
Публичное API:
- `spa_stepm(l_bench, L, idx) -> dict`: `t` (K,) студентизированные √n·d̄_k/ω̂_k (ω̂_k² — бутстрап-дисперсия √n·d̄*_k; ω̂ = 0 ⇒ t = 0); `T = max(0, max t)`; `p_l`, `p_c`, `p_u` — SPA с центровкой μ̂ = min(d̄,0) / d̄·1{t ≤ −√(2 ln ln n)} / 0 (p_l ≤ p_c ≤ p_u);
  основной p SPA этапа — `p_c` (рекомендация Hansen); `p_step` (K,) — скорректированные p StepM (step-down, максимум по оставшимся), `reject` (K,) — p_step < α.
  p = доля бутстрап-значений max(0, max_k Z*) ≥ T. NaN в потерях — ValueError (фильтр до вызова).
- `check_spa(sigma_data, lvl, n) -> list` замечаний (CLI `--check-spa` модуля `a_eval`; σ_LR разностей W5-G − S-A2 из данных ≈ 0,08): синтетика с известным ответом — кандидат лучше на 4·SE (≈ 3,2 % QLIKE базы, z = 4) ⇒ p_c < 0,05 и StepM отвергает ровно его;
  все кандидаты равны базе ⇒ p_c > 0,1; нулевая гипотеза со случайными разностями: доля p_c < 0,05 за 40 повторов ≤ 0,15; мощность при z = 4 ≥ 0,6.
  Буквальное условие спецификации (лучше на 1 % при σ_LR из данных ⇒ p < 0,05) недостижимо при n = 2198: z ≈ 1,26, мощность ≈ 0,3 — ABANDON E.G2.1, число печатается."""
from __future__ import annotations

import math

import numpy as np

from models_v2.s18 import a_eval_stat as S


def spa_stepm(l_bench, L, idx: np.ndarray, alpha: float = S.ALPHA) -> dict:
    lb, L = np.asarray(l_bench, float), np.asarray(L, float)
    if L.ndim == 1:
        L = L[:, None]
    if not (np.isfinite(lb).all() and np.isfinite(L).all()):
        raise ValueError("NaN или inf в потерях: фильтр обязателен до SPA / StepM")
    d = lb[:, None] - L
    n, K = d.shape
    dbar = d.mean(axis=0)
    bm = np.column_stack([d[:, k][idx].mean(axis=1) for k in range(K)])
    om = np.sqrt(n * bm.var(axis=0, ddof=1))
    ok = om > 1e-15
    scale = np.where(ok, om, 1.0)
    t = np.where(ok, math.sqrt(n) * dbar / scale, 0.0)
    T = max(0.0, float(t.max()))
    thr = -math.sqrt(2.0 * math.log(math.log(n)))
    mu = dict(l=np.minimum(dbar, 0.0), c=np.where(t <= thr, dbar, 0.0), u=np.zeros(K))
    out = dict(t=t, T=T, dbar=dbar, omega=om)
    for k, m in mu.items():
        Z = np.where(ok, math.sqrt(n) * (bm - dbar + m) / scale, 0.0)
        out["p_" + k] = float(np.mean(np.maximum(Z.max(axis=1), 0.0) >= T - 1e-12))
    out["p_step"] = _stepm(np.where(ok, math.sqrt(n) * (bm - dbar) / scale, 0.0), t)
    out["reject"] = out["p_step"] < alpha
    return out


def _stepm(Z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Step-down: на каждом шаге критическая статистика — максимум Z* по ещё не отвергнутым; p монотонны."""
    rem, adj, run = list(range(len(t))), np.zeros(len(t)), 0.0
    for k in np.argsort(-t, kind="stable"):
        run = max(run, float(np.mean(Z[:, rem].max(axis=1) >= t[k] - 1e-12)))
        adj[k] = run
        rem.remove(int(k))
    return adj


def _synth(rng, n: int, sig: float, eff: float, K: int, lvl: float):
    """База — положительная (≈ lvl); кандидат 0 лучше на eff·lvl, остальные равны в среднем; разности — AR(1), σ_LR ≈ sig."""
    base = lvl * rng.chisquare(1, n)
    phi = 0.3
    e = rng.normal(0.0, sig * (1 - phi), (n, K))
    for t in range(1, n):
        e[t] += phi * e[t - 1]
    L = base[:, None] + e
    L[:, 0] -= eff * lvl
    return base, L


def check_spa(sigma_data: float | None = None, lvl: float = 0.2155, n: int = 2198) -> list:
    """Буквальное условие спецификации («лучше на 1 %» при σ_LR из данных ⇒ p < 0,05) при σ_LR ≈ 0,08 и n = 2198 недостижимо: z = 1 %·QLIKE/(σ_LR/√n) ≈ 1,3, аналитическая мощность ≈ 0,35 (ABANDON E.G2.1, печатается).
    Проверяются известный ответ с обнаружимым эффектом (z = 4: 4·σ_LR/√n), нулевые случаи и калибровка уровня."""
    sig = float(sigma_data) if sigma_data else 0.02
    bad, rng, idx = [], np.random.default_rng(0), S.idx_for(n)
    se = sig / math.sqrt(n)
    eff = 4.0 * se / lvl
    base, L = _synth(rng, n, sig, eff, 3, lvl)
    r = spa_stepm(base, L, idx)
    bad += [] if r["p_c"] < 0.05 and bool(r["reject"][0]) and int(r["reject"].sum()) == 1 else ["эффект 4·SE: p_c %.4f, reject %s" % (r["p_c"], r["reject"])]
    bad += [] if spa_stepm(base, np.column_stack([base] * 3), idx)["p_c"] > 0.1 else ["равны (копии базы): p_c ≤ 0,1"]
    ps0 = np.array([spa_stepm(*_synth(np.random.default_rng(100 + i), n, sig, 0.0, 3, lvl), S.stat_idx(n, 1000, 20, i))["p_c"] for i in range(40)])
    ps1 = np.array([spa_stepm(*_synth(np.random.default_rng(200 + i), n, sig, eff, 3, lvl), S.stat_idx(n, 1000, 20, i))["p_c"] for i in range(40)])
    ps2 = np.array([spa_stepm(*_synth(np.random.default_rng(300 + i), n, sig, 0.01, 3, lvl), S.stat_idx(n, 1000, 20, i))["p_c"] for i in range(40)])
    bad += [] if np.mean(ps0 < 0.05) <= 0.15 else ["ноль: доля p_c < 0,05 = %.3f > 0,15" % np.mean(ps0 < 0.05)]
    bad += [] if np.mean(ps1 < 0.05) >= 0.6 else ["эффект 4·SE: мощность %.3f < 0,6" % np.mean(ps1 < 0.05)]
    print("check-spa: σ_LR %.4f, SE %.5f; эффект 4·SE = %.2f %% QLIKE базы: p_l/c/u %.4f/%.4f/%.4f, t %s, p_step %s; ноль: доля отвержений %.3f (средний p_c %.3f); эффект 4·SE: мощность %.3f; "
          "ABANDON E.G2.1 (буквально 1 %%): мощность по 40 повторам %.3f, z = %.2f" % (sig, se, 100 * eff, r["p_l"], r["p_c"], r["p_u"], np.round(r["t"], 2), np.round(r["p_step"], 4), np.mean(ps0 < 0.05),
                                                                                  ps0.mean(), np.mean(ps1 < 0.05), np.mean(ps2 < 0.05), 0.01 * lvl / se))
    return bad
