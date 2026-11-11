"""W8-B, таблицы оценки (`W8_SPEC.md` разд. 5, 9): чистые функции по массивам; ничего не пишут. Все пороги — константы ЭТОГО файла (из спецификации), не из данных.
Публичное API:
- `THR` = {У1: 0,415; У2: 0,50; У3: 0,26; У6: 0,10}, `FORMULA`; `MAIN` = (K1, K2, K3, X1, W5-G, W8-G), `REFS` = (W8-Gb, M, ZB), `BASES` = (W5-G, X1, K2, K3); `N_HOLM` = 814, `N_WIDE` = 3333.
- `losses(rv, F)`, `table(rv, tp, F)`: общая таблица (QLIKE = среднее дневных потерь l = RV/F − ln(RV/F) − 1 на одних датах).
- `stats(rv, tp, F, perm_q)`: Δ W8-G к W5-G, X1, K2, K3 (основные; 90 % блочный бутстрап, DM НУ лаг 5, Холм ТОЛЬКО по этим 4 p, N = 814 / N_wide = 3333) + справочные (W8-Gb, M, ZB) + 19 сдвигов + `perm_p`
  + строки `sens` (Δ W8-G к W5-G и X1 в h1, h2, без 2022, без пары t⁺ 2022-02-24, h2 без 2022, без наиболее влиятельного года) — показ, не условие.
- `years(rv, tp, F)`, `conc(rv, tp, F)`: вклады периодов W8-G к четырём базам, Σ = Δ (1e-12); `identity_gap(yrs)`.
- `cells(x, F8)`: вклады год/половина/окно × тип перерыва (как 1.2 `W8_SPEC.md`), базы W5-G, X1, K2, K3; -> (DataFrame, максимум нарушения тождеств).
- `mcs(rv, F)`; `beta_table(w8g)`: путь β_{m,G} по месяцам, `n_G` пар в пуле, флаги включения.
- `guard(tbl, yrs, conc, pp)`: основной критерий и У1–У6 из таблиц; `decide(g)` — «третья контрольная версия» (все выполнены) или «не прошёл: …»; X1 и W5-G остаются контрольными в любом исходе."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_w1 as W1, x1_w1_stat as SS, x1_w5_eval_t as T5
from models_v2.s16.x1_w8_diag_core import YEARS
from models_v2.s16.x1_z_fit import MIN_EXT

EVX = SS.EVX
THR = {"У1": 0.415, "У2": 0.50, "У3": 0.26, "У6": 0.10}
FORMULA = {"У1": "порог W5 У1 (1,645 · SE_h1(ZB − X1) = 1,645 · 0,2525 %)", "У2": "порог W5 У2 (потолок W4, 0,50 %)", "У3": "порог W5 У3 (1,645 · SE_2018 = 1,645 · 0,1567 %)",
           "У6": "p по 19 сдвигам ≤ 0,10 (не больше одного сдвига с QLIKE ≤ W8-G)"}
MAIN, REFS, BASES = ("K1", "K2", "K3", "X1", "W5-G", "W8-G"), ("W8-Gb", "M", "ZB"), ("W5-G", "X1", "K2", "K3")
N_HOLM, N_WIDE = 814, 3333
H1, H2 = T5.H1, T5.H2
TYPES = ("weekend", "hol1", "long", "worksat", "other", "none")
losses, masks, perm_p, identity_gap = T5.losses, T5.masks, T5.perm_p, T5.identity_gap


def table(rv, tp, F) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    return pd.DataFrame([dict(role="main" if k in MAIN else "reference", series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=float(Q.ql(rv, F[k]).mean()))
                         for k in MAIN + REFS])


def _delta(l_g, l_x, keep) -> float:
    return 100.0 * float(l_g[keep].sum() - l_x[keep].sum()) / float(l_x[keep].sum())


def sens(rv, tp, F) -> pd.DataFrame:
    """Только показ: срезы сохранённых потерь, β не переоценивается."""
    ls, tp = losses(rv, F), pd.DatetimeIndex(tp)
    y, rows = tp.year.to_numpy(), []
    for b in ("W5-G", "X1"):
        g, x = ls["W8-G"], ls[b]
        cy = {int(k): float((g[y == k] - x[y == k]).sum()) for k in np.unique(y)}
        top = max(cy, key=lambda k: abs(cy[k]))
        var = [("все даты", np.ones(len(g), bool)), ("h1 (t⁺ 2018–2021)", y <= 2021), ("h2 (t⁺ 2022–2026)", y >= 2022), ("без 2022", y != 2022), ("без пары t⁺ = 2022-02-24", tp != pd.Timestamp("2022-02-24")),
               ("h2 без 2022", y >= 2023), ("без наиболее влиятельного года (%d)" % top, y != top)]
        rows += [dict(kind="sens", hybrid="W8-G", base=b, variant=v, n=int(k.sum()), qlike_h=float(g[k].mean()), qlike_b=float(x[k].mean()), delta_pct=_delta(g, x, k), verdict="показ, не условие") for v, k in var]
    return pd.DataFrame(rows)


def stats(rv, tp, F, perm_q) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    t = pd.DataFrame([T5._row("main", "W8-G", b, rv, tp, F) for b in BASES] + [T5._row("reference", "W8-G", b, rv, tp, F) for b in REFS])
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    m = (t["kind"] == "main").to_numpy()
    t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = EVX.holm_both(t.loc[m, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    t["verdict"] = [SS.verdict(d, p) if k == "main" else "справочно, критериев не получает" for d, p, k in zip(t["delta_pct"], t["p_holm_N"], t["kind"])]
    t["n_holm"], t["n_holm_wide"] = N_HOLM, N_WIDE
    q_g = float(Q.ql(rv, F["W8-G"]).mean())
    pm = pd.DataFrame([dict(kind="perm", hybrid="сдвиг %d" % (252 + 100 * j), base="W8-G", n=len(tp), qlike_h=float(q), qlike_b=q_g) for j, q in enumerate(perm_q)]
                      + [dict(kind="perm_p", hybrid="p по сдвигам: (1 + число сдвигов с QLIKE_j ≤ QLIKE(W8-G)) / %d" % (1 + len(perm_q)), base="W8-G", n=len(tp), qlike_h=q_g, p_one=perm_p(q_g, perm_q))])
    return pd.concat([t, pm, sens(rv, tp, F)], ignore_index=True)


def years(rv, tp, F) -> pd.DataFrame:
    ls, mk, rows = losses(rv, F), masks(tp), []
    for k in MAIN + REFS:
        rows += [dict(series=k, base="-", kind="half" if p in (H1, H2) else "period", period=p, n=int(m.sum()), q_series=float(ls[k][m].mean()), q_base=np.nan, delta_pct=np.nan, contrib_pct=np.nan, delta_total_pct=np.nan)
                 for p, m in mk.items()]
    for b in BASES:
        c, tot = SS.contrib(ls["W8-G"], ls[b], mk), 100.0 * float(ls["W8-G"].mean() - ls[b].mean()) / float(ls[b].mean())
        rows += [dict(series="W8-G", base=b, kind="half" if p in (H1, H2) else "period", period=p, n=int(m.sum()), q_series=float(ls["W8-G"][m].mean()), q_base=float(ls[b][m].mean()),
                      delta_pct=100.0 * float(ls["W8-G"][m].mean() / ls[b][m].mean() - 1.0), contrib_pct=c[p], delta_total_pct=tot) for p, m in mk.items()]
    return pd.DataFrame(rows)


def conc(rv, tp, F) -> pd.DataFrame:
    ls, mk, out = losses(rv, F), SS.period_masks(tp), []
    for b in BASES:
        c, tot = SS.contrib(ls["W8-G"], ls[b], mk), 100.0 * float(ls["W8-G"].mean() - ls[b].mean()) / float(ls[b].mean())
        out += SS.conc_rows("W8-G", b, ls, mk, c, tot)
    return pd.DataFrame(out)


def cells(x, F8) -> tuple:
    """Вклад в п. п. QLIKE базы: Σ(l_W8G − l_база)/(n·Q_база) по клетке; тождества: Σ по типам = Δ периода, Σ по годам = Δ окна."""
    rv, n = x["rv"].to_numpy(float), len(x)
    l8, lb = Q.ql(rv, F8), {b: x["l_" + {"W5-G": "W5G"}.get(b, b)].to_numpy() for b in BASES}
    c_app, u = np.log(np.asarray(F8, float) / x["F"].to_numpy(float)), x["u"].to_numpy()
    pers = [(y, (x["yr"] == y).to_numpy()) for y in YEARS] + [("h1", (x["half"] == "h1").to_numpy()), ("h2", (x["half"] == "h2").to_numpy()), ("M", np.ones(n, bool))]
    rows, chk = [], []
    for b in BASES:
        QM = float(lb[b].mean())
        for p, pm in pers:
            Qp, n_p, tot = float(lb[b][pm].mean()), int(pm.sum()), 0.0
            for t in TYPES + ("all_gap", "all"):
                tm = pm & ((x["type"] == t).to_numpy() if t in TYPES else ((x["gap"] == 1).to_numpy() if t == "all_gap" else np.ones(n, bool)))
                k = int(tm.sum())
                sd = float((l8[tm] - lb[b][tm]).sum())
                r = dict(base=b, period=p, type=t, n=k, beta_app=float(c_app[tm].mean()) if k else np.nan, beta_expost=float(np.log(np.mean(np.exp(u[tm])))) if k else np.nan,
                         Q_base=float(lb[b][tm].mean()) if k else np.nan, Q_W8G=float(l8[tm].mean()) if k else np.nan,
                         delta_pct=100.0 * float(l8[tm].mean() / lb[b][tm].mean() - 1.0) if k else np.nan, contrib_pp_M=sd / (n * QM) * 100, contrib_pp_period=sd / (n_p * Qp) * 100)
                rows.append(r)
                tot += r["contrib_pp_period"] if t in TYPES else 0.0
                chk += [abs(tot - r["delta_pct"])] if t == "all" else []
        a = [r for r in rows if r["base"] == b and r["type"] == "all"]
        chk.append(abs(sum(r["contrib_pp_M"] for r in a if r["period"] in YEARS) - [r["delta_pct"] for r in a if r["period"] == "M"][0]))
    return pd.DataFrame(rows), max(chk)


def mcs(rv, F) -> pd.DataFrame:
    return W1.mcs90(rv, {k: F[k] for k in ("W8-G", "W5-G", "X1", "ZB", "K2", "K3", "M")}).reset_index()


def beta_table(w8g: pd.DataFrame) -> pd.DataFrame:
    m = w8g.dropna(subset=["refit_date"]).drop_duplicates("refit_date")
    t = pd.DataFrame({"refit_date": m["refit_date"].dt.date.astype(str).to_numpy(), "beta_W5G": m["beta"].to_numpy()})
    for G in ("T1", "T2", "T3"):
        t["b_" + G], t["n_" + G], t["incl_" + G] = m["b_" + G].to_numpy(), m["n_" + G].to_numpy(), (m["n_" + G] >= MIN_EXT).astype(int).to_numpy()
    return t


def _g(rows, i, cond, v, thr, f, ok) -> None:
    rows.append(dict(id=i, condition=cond, value=v, threshold=thr, formula=f, passed=bool(ok)))


def guard(tbl, yrs, cnc, pp: float) -> pd.DataFrame:  # xtask: длинная — перечень условий У1–У6 в одном месте
    """Основной критерий и У1–У6 по таблицам; пороги — `THR`. У5: Δ(W8-G к W5-G) без периода с наименьшим вкладом ≤ 0."""
    q = tbl.set_index("series")["qlike"]
    w = yrs[(yrs["series"] == "W8-G") & (yrs["base"] == "W5-G")].set_index("period")
    dx = float(yrs[(yrs["series"] == "W8-G") & (yrs["base"] == "X1")]["delta_total_pct"].iloc[0])
    d_all = float(w["delta_total_pct"].iloc[0])
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = float(cnc[(cnc["base"] == "W5-G") & (cnc["item"] == "loyo") & (cnc["key"] == best)]["value"].iloc[0])
    rows = []
    _g(rows, "основной", "Δ(W8-G к W5-G) < 0 и Δ(W8-G к X1) < 0 на окне M, %", "W5-G=%+.4f; X1=%+.4f" % (d_all, dx), 0.0, "обе строго меньше нуля", d_all < 0 and dx < 0)
    for i, p in (("У1", H1), ("У2", H2), ("У3", "2018")):
        v = float(w.at[p, "delta_pct"])
        _g(rows, i, "Δ(W8-G к W5-G) в %s ≤ порога, %%" % p, v, THR[i], FORMULA[i], v <= THR[i])
    _g(rows, "У4", "QLIKE(W8-G) < QLIKE(W8-Gb) на окне M", "G=%.6f; Gb=%.6f" % (q["W8-G"], q["W8-Gb"]), "—", "строгое неравенство", q["W8-G"] < q["W8-Gb"])
    _g(rows, "У5", "Δ(W8-G к W5-G) без лучшего периода (%s) ≤ 0, %%" % best, loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)
    _g(rows, "У6", "p по 19 сдвигам ≤ порога", pp, THR["У6"], FORMULA["У6"], pp <= THR["У6"])
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    try:
        return "%+.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def decide(g: pd.DataFrame) -> str:
    """Решение выводится только из строк условий (в т. ч. прочитанных из `W8_GUARD.csv`: значения могут быть строками)."""
    c = g[~g["id"].isin(("справка", "решение"))]
    bad = c[~c["passed"].astype(bool)]
    if bad.empty:
        return "третья контрольная версия: W8-G (решение разработки, не «подтверждено»); X1 и W5-G остаются контрольными версиями"
    return "W8-G не прошёл: " + "; ".join("%s (%s)" % (r.id, _fmt(r.value)) for r in bad.itertuples()) + "; X1 и W5-G остаются контрольными версиями"
