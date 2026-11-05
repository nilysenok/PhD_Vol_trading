"""D2: статистика финалистов (DM по периодам, MCS, блочный бутстрап, Холм N=106/116), топ-20, LOYO, нижний дециль, режимы, таблица финалистов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s5.c import ctx_c, mix, tab_c
from models_v2.s5.c import stages as cstages
from models_v2.s5.d import ctx_d, diag_d, robust_d, select_d, stages_d, stats_d, tab_d

BASES = (("HARJ", "HAR-J(б)"), ("B2", "Б-2"))


def sigmas(C: dict, best: str) -> tuple:
    """-> (ctx, {rid: σ²}) для финалистов, справочной строки и баз (HARJ, B2); seed 42."""
    ctx = mix.context(C["w"].F, C["w"].grid, C["ctrl"])
    S = stages_d.Src(C)
    sig = {r: robust_d.sigma(r, best, ctx, robust_d.bundle(S, robust_d.needs(r, best))) for r in robust_d.finalists(best) + [robust_d.REF_ROW]}
    sig["HARJ"], sig["B2"] = C["ctrl"]["s2_harj"].to_numpy(float), C["ctrl"]["s2_b2"].to_numpy(float)
    return ctx, sig


def test_rows(rv, sig, td, rids, idx) -> pd.DataFrame:
    rec = []
    for r in rids:
        for bk, bn in BASES:
            for e in stats_d.dm_periods(rv, sig[bk], sig[r], td):
                rec.append({"test": "DM", "row_id": r, "base": bn, "period": e["period"], "delta": e["d"], "stat": e["dm"], "p": e["p"], "n": e["n"]})
            bd = stats_d.boot_delta(metrics.qlike(rv, sig[r]) - metrics.qlike(rv, sig[bk]), idx)
            rec.append({"test": "boot", "row_id": r, "base": bn, "period": "весь период", "delta": bd["mean"], "lo95": bd["lo95"], "hi95": bd["hi95"],
                        "p": bd["p_boot"], "n": len(rv)})
    return pd.DataFrame(rec)


def mcs_rows(rv, sig, td, rids) -> tuple:
    names = rids + ["HARJ", "B2"]
    L = np.column_stack([metrics.qlike(rv, sig[k]) for k in names])
    y = pd.Series(td).dt.year.to_numpy()
    rec, res_all = [], None
    for lab, m in (("весь период", np.ones(len(y), bool)), ("2017–2021 (справочно)", y <= 2021), ("2022–2026 (справочно)", y >= 2022)):
        res = stats_d.mcs(L[m], stats_d.block_idx(int(m.sum())))
        res_all = res if res_all is None else res_all
        rec += [{"test": "MCS", "row_id": k, "base": "", "period": lab, "delta": float(L[m][:, j].mean()), "p": float(res["p"][j]), "in_mcs": bool(res["in"][j]),
                 "n": int(m.sum())} for j, k in enumerate(names)]
    return pd.DataFrame(rec), dict(zip(names, res_all["in"]))


def holm_rows(df: pd.DataFrame, rids) -> pd.DataFrame:
    df = df.assign(p_holm_N106=np.nan, p_holm_N116=np.nan)
    for t, per in (("DM", "весь период"), ("DM", "без 24.02.2022"), ("boot", "весь период")):
        for _, bn in BASES:
            m = (df["test"] == t) & (df["period"] == per) & (df["base"] == bn) & df["row_id"].isin(rids)
            df.loc[m, "p_holm_N106"] = stats_d.holm_n(df.loc[m, "p"].to_numpy(), tab_d.FAMILY)
            df.loc[m, "p_holm_N116"] = stats_d.holm_n(df.loc[m, "p"].to_numpy(), tab_d.FAMILY_REF)
    return df


def diag_tables(C, ctx, sig, rids, td) -> tuple:
    rv, g = ctx["rv"], C["w"].grid
    t20, ly, lw, rg = [], [], [], []
    for r in rids:
        t = diag_d.top20(rv, sig[r], sig["HARJ"], td)
        t20.append(t.assign(row_id=r, **t.attrs))
        full_h, full_b = (float((metrics.qlike(rv, sig[r]) - metrics.qlike(rv, sig[k])).mean()) for k in ("HARJ", "B2"))
        y = diag_d.loyo_years(rv, sig[r], sig["HARJ"], sig["B2"], td)
        ly.append(y.assign(row_id=r, d_harj_full=full_h, d_b2_full=full_b, sign_kept_harj=np.sign(y["d_harj"]) == np.sign(full_h), sign_kept_b2=np.sign(y["d_b2"]) == np.sign(full_b)))
    mask = diag_d.low_rv_mask(np.exp(C["w"].F.X[:, 1]), g)
    s2 = {"harj": sig["HARJ"], "b2": sig["B2"], **{r: sig[r] for r in rids}}
    lw = diag_d.low_rv_table(rv, s2, mask)
    qw = ctx["qw"]
    groups = {"Низкая RV (квинтили 0–1)": qw <= 1, "Средняя RV (квинтиль 2)": qw == 2, "Высокая RV (квинтили 3–4)": qw >= 3, "Стресс (Г4)": ctx["st_g4"].astype(bool)}
    for lab, m in groups.items():
        for k, v in s2.items():
            rg.append({"group": lab, "model": k, "n": int(m.sum()), "qlike": float(metrics.qlike(rv, v)[m].mean())})
    return pd.concat(t20), pd.concat(ly), lw, pd.DataFrame(rg)


def fin_table(C, best, sig, flags, mcs_in, ctx) -> pd.DataFrame:
    """Строки финалистов: внутренний критерий (вид), флаги, периоды, p Холма; источник — C/D rows.json или пересчёт (b2_f4)."""
    rows = {**ctx_c.load_json("rows.json", {}), **ctx_d.load_json("rows.json", {})}
    rec = []
    for r in robust_d.finalists(best) + [robust_d.REF_ROW]:
        if r == "b2_f4":
            f4 = cstages.daily(C, None, "f4", C["i2"])[0]
            x = tab_c.build_row(r, f4, C["ctrl"], f4, "loyo", ctx_c.load_json("crit_f4.json", 0.0), cstages.crit_b2(), False, "feature_engineering", "", None)
        else:
            x = dict(rows[r])
        fl = flags["flags"][r]
        rec.append({**{k: x[k] for k in ("row_id", "q_int_kind", "q_int", "q_int_b2", "perspective", "qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025",
                                         "qlike_2026", "qlike_no_240222", "d_harj_no_240222")}, "seed_ok": fl["seed_ok"], "n_seed_ok": fl["n_ok"], "plateau_ok": fl["plateau_ok"],
                    "mcs_in": bool(mcs_in.get(r, False)), "eval_selected": r in select_d.EVAL_SELECTED})
    return pd.DataFrame(rec)


def run(C) -> None:
    best = ctx_d.load_json("best.json", {})["best"]
    flags = ctx_d.load_json("flags.json", {})
    ctx, sig = sigmas(C, best)
    td, rv = C["ctrl"]["target_date"], ctx["rv"]
    rids = robust_d.finalists(best)
    idx = stats_d.block_idx(len(rv))
    mc, mcs_in = mcs_rows(rv, sig, td, rids)
    st = holm_rows(pd.concat([test_rows(rv, sig, td, rids, idx), mc], ignore_index=True), rids)
    ctx_d.write(st, common.REP / "STATISTICAL_TESTS.csv", "DM (Ньюи–Уэст 5), MCS T_max 90 %%, блочный бутстрап (блок %d, %d, seed %d); Холм N=%d (справочно %d); подпериоды — сырые p" %
                (stats_d.BLOCK, stats_d.B, stats_d.SEED, tab_d.FAMILY, tab_d.FAMILY_REF))
    t20, ly, lw, rg = diag_tables(C, ctx, sig, rids, td)
    ctx_d.write(t20, common.REP / "D_TOP20.csv", "20 дней с наибольшей |Δ потерь| к HAR-J(б); доли в Σ Δ и Σ|Δ| — в колонках sum_*")
    ctx_d.write(ly, common.REP / "D_LOYO.csv", "описание без каждого года; sign_kept — знак Δ как у полной выборки")
    ctx_d.write(lw, common.REP / "D_LOWRV.csv", "нижний дециль RV_w(d) по 252 предыдущим (без текущего дня)")
    ctx_d.write(rg, common.REP / "D_REGIMES.csv", "QLIKE по режимам PIT: квинтили RV_w по 252 предыдущим; стресс Г4")
    fin = fin_table(C, best, sig, flags, mcs_in, ctx)
    ctx_d.write(fin, ctx_d.DATA / "fin_table.csv", "финалисты D2")
    top = select_d.top3(fin)
    ctx_d.save_json("top3.json", {"top3": list(top["row_id"]), "excluded": {r.row_id: [k for k in ("perspective", "seed_ok", "plateau_ok", "mcs_in") if not r[k]]
                                                                        for _, r in fin.iterrows() if r.row_id not in set(top["row_id"])}})
    print("G9: тройка", list(top["row_id"]), "| MCS(весь период):", mcs_in)
