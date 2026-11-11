"""W5-B, таблицы оценки (раздел 5, 6, 11.6 `W5_SPEC.md`): чистые функции по массивам; ничего не пишут. Все пороги — константы ЭТОГО файла (из спецификации), не из данных.
Публичное API:
- `THR` = {У1: 0,415; У2: 0,50; У3: 0,26} (%), `FORMULA` — формулы порогов; `MAIN` = (K1, K2, K3, X1, W5-G), `REFS` = (M, W5-Ga, W5-Gb, W5-GK2); `N_HOLM` = 798, `N_WIDE` = 3301.
- `table(rv, tp, F)`: общая таблица (QLIKE = среднее дневных потерь на одних датах). `stats(rv, tp, F, perm_q)`: Δ W5-G к X1, K2, K3 (основные, Холм только по ним, N = 798) и справочные строки + перестановки.
- `masks(tp)`: периоды 2018…2025, 2026_до_27.02, 2026_F, половины h1_2018-2021 / h2_2022-2026. `years(rv, tp, F)`, `conc(rv, tp, F)`: вклады лет, Σ = Δ (1e-12), концентрация.
- `gapdays(rv, tp, F, gap, delta)`, `sens(rv, tp, F)`, `mcs(rv, F)`; `perm_p(q_g, perm_q)` = (1 + #{q_j ≤ q_g}) / (1 + число сдвигов).
- `guard(tbl, yrs, conc, perm_p)`: основной критерий и У1–У5 из таблиц; `decide(g)` — решение «контрольная точка» (все условия выполнены) или «X1 остаётся» с названием невыполненных."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_t0 as T0, x1_w1 as W1, x1_w1_stat as SS

ST, EVX = T0.ST, T0.EVX
THR = {"У1": 0.415, "У2": 0.50, "У3": 0.26}
FORMULA = {"У1": "1,645 · SE_h1(ZB − X1) = 1,645 · 0,2525 %", "У2": "min(1,645 · SE_h2 = 2,57 %; 0,50 % W4)", "У3": "1,645 · SE_2018(ZB − X1) = 1,645 · 0,1567 %"}
MAIN, REFS = ("K1", "K2", "K3", "X1", "W5-G"), ("M", "W5-Ga", "W5-Gb", "W5-GK2")
N_HOLM, N_WIDE = 798, 3301
H1, H2 = "h1_2018-2021", "h2_2022-2026"


def losses(rv, F) -> dict:
    return {k: Q.ql(rv, v) for k, v in F.items()}


def table(rv, tp, F) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    return pd.DataFrame([dict(role="main" if k in MAIN else "reference", series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=float(Q.ql(rv, F[k]).mean()))
                         for k in MAIN + REFS])


def _row(kind, h, b, rv, tp, F) -> dict:
    qh, qb = Q.ql(rv, F[h]), Q.ql(rv, F[b])
    ci, dm = EVX.delta_ci(qh, qb, float(qb.mean()), b=ST.B, block=ST.BLOCK, seed=ST.SEED), EVX.dm_one(rv, F[b], F[h])
    return dict(kind=kind, hybrid=h, base=b, n=len(rv), first=str(tp.min().date()), last=str(tp.max().date()), qlike_h=float(qh.mean()), qlike_b=float(qb.mean()), delta_pct=ci["delta_pct"],
                lo90=ci["lo90"], hi90=ci["hi90"], dm_stat=dm["stat"], p_one=dm["p_one"])


def perm_p(q_g: float, perm_q) -> float:
    perm_q = np.asarray(perm_q, float)
    return float((1 + (perm_q <= q_g).sum()) / (1 + len(perm_q)))


def stats(rv, tp, F, perm_q) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    rows = [_row("main", "W5-G", b, rv, tp, F) for b in ("X1", "K2", "K3")] + [_row("reference", "W5-G", b, rv, tp, F) for b in ("M", "W5-Ga", "W5-Gb")] + [_row("reference", "W5-GK2", "K2", rv, tp, F)]
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    m = (t["kind"] == "main").to_numpy()
    t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = EVX.holm_both(t.loc[m, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    t["verdict"] = [SS.verdict(d, p) if k == "main" else "справочно, критериев не получает" for d, p, k in zip(t["delta_pct"], t["p_holm_N"], t["kind"])]
    t["n_holm"], t["n_holm_wide"] = N_HOLM, N_WIDE
    q_g = float(Q.ql(rv, F["W5-G"]).mean())
    pm = pd.DataFrame([dict(kind="perm", hybrid="сдвиг %d" % (252 + 100 * j), base="W5-G", n=len(tp), qlike_h=float(q), qlike_b=q_g) for j, q in enumerate(perm_q)]
                      + [dict(kind="perm_p", hybrid="p по перестановкам: (1 + число сдвигов с QLIKE_j ≤ QLIKE(W5-G)) / %d" % (1 + len(perm_q)), base="W5-G", n=len(tp), qlike_h=q_g, p_one=perm_p(q_g, perm_q))])
    return pd.concat([t, pm], ignore_index=True)


def masks(tp) -> dict:
    m = SS.period_masks(tp)
    y = pd.DatetimeIndex(tp).year.to_numpy()
    return dict(m) | {H1: y <= 2021, H2: y >= 2022}


def years(rv, tp, F) -> pd.DataFrame:
    ls, mk, rows = losses(rv, F), masks(tp), []
    for k in MAIN + REFS:
        rows += [dict(series=k, base="-", kind="half" if p in (H1, H2) else "period", period=p, n=int(m.sum()), q_series=float(ls[k][m].mean()), q_base=np.nan, delta_pct=np.nan, contrib_pct=np.nan, delta_total_pct=np.nan) for p, m in mk.items()]
    for b in ("X1", "K2", "K3"):
        c, tot = SS.contrib(ls["W5-G"], ls[b], mk), 100.0 * float(ls["W5-G"].mean() - ls[b].mean()) / float(ls[b].mean())
        rows += [dict(series="W5-G", base=b, kind="half" if p in (H1, H2) else "period", period=p, n=int(m.sum()), q_series=float(ls["W5-G"][m].mean()), q_base=float(ls[b][m].mean()),
                      delta_pct=100.0 * float(ls["W5-G"][m].mean() / ls[b][m].mean() - 1.0), contrib_pct=c[p], delta_total_pct=tot) for p, m in mk.items()]
    return pd.DataFrame(rows)


def conc(rv, tp, F) -> pd.DataFrame:
    ls, mk, out = losses(rv, F), SS.period_masks(tp), []
    for b in ("X1", "K2", "K3"):
        c, tot = SS.contrib(ls["W5-G"], ls[b], mk), 100.0 * float(ls["W5-G"].mean() - ls[b].mean()) / float(ls[b].mean())
        out += SS.conc_rows("W5-G", b, ls, mk, c, tot)
    return pd.DataFrame(out)


def identity_gap(yrs: pd.DataFrame) -> float:
    """max |Σ вкладов периодов − Δ| по основам; то же для половин."""
    p = yrs[yrs["base"] != "-"]
    return float(max(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) for _, g in p.groupby(["base", "kind"])))


def gapdays(rv, tp, F, gap, delta) -> pd.DataFrame:
    ls, n = losses(rv, F), len(rv)
    sets = [("gap^PIT = 1", gap == 1), ("gap^PIT = 0", gap == 0), ("Δ_d = 1 (факт)", delta == 1), ("Δ_d = 2 (факт)", delta == 2), ("Δ_d ≥ 3 (факт)", delta >= 3), ("Δ_d = 0 (факт)", delta == 0)]
    rows = []
    for name, m in sets:
        qx, qg = float(ls["X1"][m].mean()), float(ls["W5-G"][m].mean())
        rows.append(dict(subset=name, n=int(m.sum()), share=float(m.mean()), q_x1=qx, q_w5g=qg, delta_pct=100.0 * (qg / qx - 1.0), contrib_pct=100.0 * float((ls["W5-G"][m] - ls["X1"][m]).sum()) / (n * float(ls["X1"].mean())),
                         max_abs_loss_diff=float(np.abs(ls["W5-G"][m] - ls["X1"][m]).max())))
    return pd.DataFrame(rows)


def _delta(l_g, l_x, keep) -> float:
    return 100.0 * float(l_g[keep].sum() - l_x[keep].sum()) / float(l_x[keep].sum())


def sens(rv, tp, F) -> pd.DataFrame:
    """Только диагностика: срезы сохранённых потерь, β не переоценивается."""
    ls, tp = losses(rv, F), pd.DatetimeIndex(tp)
    g, x, y = ls["W5-G"], ls["X1"], tp.year.to_numpy()
    cy = {int(k): float((g[y == k] - x[y == k]).sum()) for k in np.unique(y)}
    top = max(cy, key=lambda k: abs(cy[k]))
    trim = np.abs(g - x) < np.sort(np.abs(g - x))[len(g) - int(np.ceil(0.01 * len(g)))]
    var = [("все даты (основной результат)", "—", np.ones(len(g), bool)), ("без пары t⁺ = 2022-02-24", "2022-02-24", tp != pd.Timestamp("2022-02-24")),
           ("без наиболее влиятельного года (наибольший модуль вклада в Δ к X1)", str(top), y != top), ("без 1 % дат с наибольшим модулем разности потерь W5-G и X1", "%d дат" % (len(g) - int(trim.sum())), trim)]
    return pd.DataFrame([dict(variant=v, excluded=e, n=int(k.sum()), q_x1=float(x[k].mean()), q_w5g=float(g[k].mean()), delta_pct=_delta(g, x, k)) for v, e, k in var])


def mcs(rv, F) -> pd.DataFrame:
    return W1.mcs90(rv, {k: F[k] for k in ("W5-G", "X1", "ZB", "K2", "K3", "M")}).reset_index()


def _g(rows, i, cond, v, thr, f, ok) -> None:
    rows.append(dict(id=i, condition=cond, value=v, threshold=thr, formula=f, passed=bool(ok)))


def guard(tbl, yrs, cnc, pp: float) -> pd.DataFrame:
    """Основной критерий и У1–У5 по таблицам; пороги — `THR`. У5: Δ без периода с наибольшим выигрышем (наименьший вклад) ≤ 0."""
    q = tbl.set_index("series")["qlike"]
    w = yrs[(yrs["series"] == "W5-G") & (yrs["base"] == "X1")].set_index("period")
    d_all = float(w["delta_total_pct"].iloc[0])
    per = w[w["kind"] == "period"]["contrib_pct"]
    best = per.idxmin()
    loyo = float(cnc[(cnc["base"] == "X1") & (cnc["item"] == "loyo") & (cnc["key"] == best)]["value"].iloc[0])
    rows = []
    _g(rows, "основной", "Δ(W5-G к X1) на окне M < 0, %", d_all, 0.0, "строго меньше нуля", d_all < 0)
    for i, p, f in (("У1", H1, "≤"), ("У2", H2, "≤"), ("У3", "2018", "≤")):
        v = float(w.at[p, "delta_pct"])
        _g(rows, i, "Δ(W5-G к X1) в %s ≤ порога, %%" % p, v, THR[i], FORMULA[i], v <= THR[i])
    _g(rows, "У4", "QLIKE(W5-G) < QLIKE(W5-Ga) и < QLIKE(W5-Gb) на окне M", "G=%.6f; Ga=%.6f; Gb=%.6f" % (q["W5-G"], q["W5-Ga"], q["W5-Gb"]), "—", "строгое неравенство с обоими контролями",
       q["W5-G"] < q["W5-Ga"] and q["W5-G"] < q["W5-Gb"])
    _g(rows, "У5", "Δ(W5-G к X1) без лучшего периода (%s) ≤ 0, %%" % best, loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)
    _g(rows, "справка", "p по перестановкам (показ, не условие)", pp, "—", "(1 + число сдвигов с QLIKE_j ≤ QLIKE(W5-G)) / 20", True)
    return pd.DataFrame(rows)


def decide(g: pd.DataFrame) -> str:
    c = g[~g["id"].isin(("справка", "решение"))]
    bad = c[~c["passed"]]
    if bad.empty:
        return "новая контрольная точка: W5-G заменяет X1 (решение разработки, не «подтверждено»)"
    return "X1 остаётся контрольной точкой; W5-G не прошёл: " + "; ".join("%s (%s)" % (r.id, r.value if isinstance(r.value, str) else "%+.4f" % r.value) for r in bad.itertuples())
