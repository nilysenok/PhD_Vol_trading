"""S-L, L1 и L2 (`S_D_SPEC.md` разд. 7.2; объявления О-1…О-6 — `<sp>/s_l/GATES.md`). Обучения нет: только чтение сохранённых рядов.
L1 — среднее min(RV/F, c_d) по PIT-группам (π, v₁, v₂ — квинтили по датам < d; s_d, gap, год — по значению) для W5-G, X1, cfg8, B_mix; c_d — Q₀,₉₉ RV/F по парам t⁺ < d.
L2 — 5×5 (π × v), удержание разброса по π внутри v (и обратно), потолок «постоянный множитель по группе» (использует исход).
Пишет `<sp>/s_l/l12.csv`. Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_l/l12.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import a_model as AM
from models_v2.s18 import a_walk as AW
from models_v2.s18 import cm_layers as CL

HERE = Path(__file__).parent
BURN, QG, QC = 252, (0.2, 0.4, 0.6, 0.8), 0.99
ROWS = []


def put(sec, ser, var, grp, item, n, value, note=""):
    ROWS.append(dict(section=sec, series=ser, var=var, group=str(grp), item=item, n=int(n), value=float(value), note=note))


def pit_quintile(v, i) -> int:
    """Квинтиль v[i] по порогам из v[:i] (даты < d; окно M отсортировано по d, d уникальны)."""
    q = np.quantile(v[:i], QG)
    return int(np.searchsorted(q, v[i], side="right")) + 1


def pit_groups(v) -> np.ndarray:
    g = np.zeros(len(v), int)
    for i in range(BURN, len(v)):
        g[i] = pit_quintile(v, i)
    return g


def pit_clip(rv, F, d, tp) -> np.ndarray:
    """c_d = Q₀,₉₉ RV/F по парам с t⁺ < d (своя пара дня d не входит); NaN, если пар < BURN."""
    r, c = rv / F, np.full(len(d), np.nan)
    for i in range(len(d)):
        m = tp < d[i]
        if m.sum() >= BURN:
            c[i] = np.quantile(r[m], QC)
    return c


def check_pit(v, rng) -> list:
    """Порча v после строки i не меняет квинтиль строки i (3 случайных дня)."""
    bad = []
    for i in rng.integers(BURN, len(v) - 1, 3):
        w = v.copy()
        w[i + 1:] = rng.normal(0, 10, len(v) - i - 1)
        bad += [] if pit_quintile(w, i) == pit_quintile(v, i) else ["строка %d" % i]
    return bad


def ql(rv, F):
    r = rv / F
    return r - np.log(r) - 1.0


def main():
    x, gap = CL.inputs()
    w5 = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    assert (w5["d"].to_numpy() == x["d"].to_numpy()).all()
    D = AM.load_data()
    dp = AW._day_pos(D, x)
    assert (pd.DatetimeIndex(D.d[dp]) == pd.DatetimeIndex(x["d"])).all()
    rv_d, rv_m = np.exp(D.lnS0[dp, 0]), np.exp(D.lnS0[dp, 2])
    d, tp, rv = x["d"].to_numpy(), x["t_plus"].to_numpy(), x["rv"].to_numpy(float)
    pi = x["pi"].to_numpy(float)
    V = {"pi": pi, "v1": np.log(rv_d / rv_m), "v2": np.log(x["M"].to_numpy(float) / x["C"].to_numpy(float))}
    SER = {"W5-G": w5["F"].to_numpy(float), "X1": x["F"].to_numpy(float), "cfg8": x["C"].to_numpy(float), "B_mix": x["M"].to_numpy(float)}
    rng = np.random.default_rng(0)
    for k, v in V.items():
        b = check_pit(v, rng)
        put("check", "-", k, "-", "pit_quintile_corrupt_after_d", 3, len(b), "; ".join(b) or "ok")
    G = {k: pit_groups(v) for k, v in V.items()}
    G["s_d"], G["gap"], G["year"] = x["s"].to_numpy(int), np.asarray(gap, int), pd.DatetimeIndex(d).year.to_numpy()
    clip = {s: pit_clip(rv, F, d, tp) for s, F in SER.items()}
    L = (np.arange(len(d)) >= BURN) & np.all([np.isfinite(c) for c in clip.values()], axis=0)
    put("L1", "-", "-", "-", "days_in_L1", L.sum(), L.sum(), "окно L1: %s…%s" % (pd.Timestamp(d[L][0]).date(), pd.Timestamp(d[L][-1]).date()))
    put("L1", "-", "-", "-", "days_burn_in", (~L).sum(), (~L).sum(), "разогрев %d строк / пар" % BURN)
    for a in ("v1", "v2"):
        put("L1", "-", "pi~" + a, "-", "corr_pearson", L.sum(), np.corrcoef(pi[L], V[a][L])[0, 1])
        put("L1", "-", "pi~" + a, "-", "corr_spearman", L.sum(), pd.Series(pi[L]).corr(pd.Series(V[a][L]), method="spearman"))
    for y in np.unique(G["year"][L]):
        m = L & (G["year"] == y)
        put("L1", "pi", "year", y, "mean_pi", m.sum(), pi[m].mean())
    for s, F in SER.items():
        r = rv / F
        rc = np.minimum(r, clip[s])
        put("L1", s, "all", "all", "mean_clipped_ratio", L.sum(), rc[L].mean())
        for var, g in G.items():
            for gv in np.unique(g[L]):
                m = L & (g == gv)
                put("L1", s, var, gv, "mean_clipped_ratio", m.sum(), rc[m].mean())
                put("L1", s, var, gv, "mean_ratio", m.sum(), r[m].mean())
                put("L1", s, var, gv, "mean_ln_ratio", m.sum(), np.log(r[m]).mean())
        for a in ("v1", "v2"):
            l2(s, a, rc, G["pi"], G[a], L)
        within_year(s, rc, G["pi"], G["year"], L)
        for grp in ("one", "pi", "v1", "v2", "pi×v1", "pi×v2"):
            ceiling(s, grp, rv, F, G, L)
    pd.DataFrame(ROWS).to_csv(HERE / "l12.csv", index=False, float_format="%.10g")
    pd.set_option("display.width", 250, "display.max_rows", 3000)
    print(pd.DataFrame(ROWS).to_string(index=False, float_format=lambda v: "%.6g" % v))


def l2(s, a, rc, gp, ga, L):
    """5×5 средних; S_π = r̄(π5) − r̄(π1); S_π|v = среднее по v-квинтилям той же разности; R = отношение. Обратно — для v."""
    T = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            m = L & (gp == i + 1) & (ga == j + 1)
            T[i, j] = rc[m].mean() if m.any() else np.nan
            put("L2", s, "pi×" + a, "pi%d|%s%d" % (i + 1, a, j + 1), "mean_clipped_ratio", m.sum(), T[i, j])
    mp = [rc[L & (gp == q)].mean() for q in (1, 5)]
    mv = [rc[L & (ga == q)].mean() for q in (1, 5)]
    S_pi, S_v = mp[1] - mp[0], mv[1] - mv[0]
    S_pi_v, S_v_pi = np.nanmean(T[4, :] - T[0, :]), np.nanmean(T[:, 4] - T[:, 0])
    R1, R2 = S_pi_v / S_pi, S_v_pi / S_v
    for it, val in (("S_pi_marginal", S_pi), ("S_pi_within_v", S_pi_v), ("R_pi_given_v", R1), ("S_v_marginal", S_v), ("S_v_within_pi", S_v_pi), ("R_v_given_pi", R2)):
        put("L2", s, "pi×" + a, "-", it, L.sum(), val)
    code = 1 if R1 >= 0.5 else (2 if R2 >= 0.5 else 0)
    put("L2", s, "pi×" + a, "-", "reading_O5", L.sum(), code, {1: "π не объясняется %s" % a, 2: "π ≈ режим %s" % a, 0: "не разделено"}[code])


def within_year(s, rc, gp, gy, L):
    """Дополнительно (не в разд. 7.2, на строку не влияет): разброс по π внутри лет 2019–2026 — не прокси ли π времени."""
    yrs = [y for y in np.unique(gy[L]) if (L & (gy == y)).sum() >= 100]
    d = [rc[L & (gy == y) & (gp == 5)].mean() - rc[L & (gy == y) & (gp == 1)].mean() for y in yrs]
    S_pi = rc[L & (gp == 5)].mean() - rc[L & (gp == 1)].mean()
    put("L2_extra", s, "pi×year", "-", "S_pi_within_year", L.sum(), float(np.nanmean(d)), "годы %d–%d, ≥ 100 дней" % (yrs[0], yrs[-1]))
    put("L2_extra", s, "pi×year", "-", "R_pi_given_year", L.sum(), float(np.nanmean(d)) / S_pi)
    for y, v in zip(yrs, d):
        put("L2_extra", s, "pi×year", y, "S_pi_in_year", (L & (gy == y)).sum(), v)


def ceiling(s, grp, rv, F, G, L):
    """Потолок: F·k_g, k_g = mean(RV/F) в группе g на днях L (исход используется) — выигрыш QLIKE, % QLIKE F на днях L."""
    if grp == "one":
        key = np.zeros(len(rv), int)
    elif "×" in grp:
        p, a = grp.split("×")
        key = G[p] * 10 + G[a]
    else:
        key = G[grp]
    Fk = F.copy()
    for gv in np.unique(key[L]):
        m = L & (key == gv)
        Fk[m] = F[m] * (rv[m] / F[m]).mean()
    base = ql(rv, F)[L].mean()
    put("L2_ceiling", s, grp, "-", "gain_pct_of_qlike", L.sum(), 100.0 * (ql(rv, Fk)[L].mean() - base) / base, "потолок: k_g по исходу, не прогноз")
    put("L2_ceiling", s, grp, "-", "n_groups", L.sum(), len(np.unique(key[L])), "потолок")


if __name__ == "__main__":
    main()
