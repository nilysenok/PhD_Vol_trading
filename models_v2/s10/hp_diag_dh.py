"""P3c, G13: R15 (г) неоднородность (LOTO против пула), (д) предсказуемость ошибки HAR-J, (е) наблюдения, (ж) регуляризация и λ = 1 (из сохранённой g, без обучения), (з) согласование рядов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s10 import hp_diag_core as C, hp_diag_data as D, hp_engine as E, hp_paths as P

CFGS = ("A1", "D2")


def _q(rv, s2) -> np.ndarray:
    return np.asarray(metrics.qlike(rv, s2))


def cause_d(df: pd.DataFrame, loto: pd.DataFrame) -> tuple:
    rows, met, aux = [], {}, []
    for cfg in sorted(loto["config"].unique()):
        m, name = cfg.split("_")
        x = loto[loto["config"] == cfg].merge(df[D.KEY + ["t_plus", "group", "rv", "k1", "%s_%s" % (m, name)]], on=D.KEY).rename(columns={"sigma2": "lo", "%s_%s" % (m, name): "pool"})
        for pop in ("M16", "SBER"):
            for w in D.WINDOWS:
                d = D.sub(x, pop, w, ["k1", "lo", "pool"])
                if d.empty:
                    continue
                dp, dk = _q(d["rv"], d["lo"]) - _q(d["rv"], d["pool"]), _q(d["rv"], d["lo"]) - _q(d["rv"], d["k1"])
                per = pd.Series(dp).groupby(d["ticker"].to_numpy()).mean()
                rows += [D.row("d", pop, w, cfg, "QLIKE_LOTO_minus_pool_mean_tickers", float(per.mean()), len(per)), D.row("d", pop, w, cfg, "frac_tickers_LOTO_worse_than_pool", float((per > 0).mean()), len(per)),
                         D.row("d", pop, w, cfg, "QLIKE_LOTO_minus_K1", float(dk.mean()), len(d))]
                aux += [{"config": cfg, "ticker": t, "window": w, "loto_minus_pool": v} for t, v in per.items()] if w in ("all", "conf") else []
                if pop == "M16" and w == "conf":
                    met["loto_" + cfg] = float(per.mean())
    return rows, met, pd.DataFrame(aux)


def cause_e(df: pd.DataFrame) -> tuple:
    rows, met = [], {}
    for pop in D.POPS:
        for w in D.WINDOWS:
            for m in D.MODS:
                for name in CFGS:
                    d = D.sub(df, pop, w, ["k1", "%s_%s" % (m, name)])
                    if len(d) < 250:
                        continue
                    r = C.r2_correction(np.log(d["rv"]) - np.log(d["k1"]), np.log(d["%s_%s" % (m, name)]) - np.log(d["k1"]))
                    rows += [D.row("e", pop, w, "%s_%s" % (m, name), k, v, len(d)) for k, v in r.items()]
                    if pop == "M16" and name == "A1" and w in ("all", "conf"):
                        met["r2_%s_%s" % (m, w)] = r["r2_mz"]
    return rows, met, None


def cause_f(df: pd.DataFrame) -> tuple:
    rows, rhos, aux = [], [], []
    for pop in ("M16", "E11"):
        for m in D.MODS:
            for name in CFGS:
                col = "%s_%s" % (m, name)
                d = D.sub(df, pop, "all", ["k1", col])
                g = pd.DataFrame({"y": d["t_plus"].dt.year, "dq": _q(d["rv"], d["k1"]) - _q(d["rv"], d[col]), "ntr": d["ntr_" + m]}).groupby("y").mean()
                aux += [{"population": pop, "config": col, "year": int(y), "gain_vs_K1": r.dq, "mean_n_train_pairs": r.ntr} for y, r in g.iterrows()]
                ex = g.drop(index=2022, errors="ignore").dropna()
                rho, rho_all = C.spearman(ex["dq"], ex["ntr"]), C.spearman(g.dropna()["dq"], g.dropna()["ntr"])
                rows += [D.row("f", pop, "all", col, "spearman_gain_vs_ntrain_ex2022", rho, len(ex)), D.row("f", pop, "all", col, "spearman_gain_vs_ntrain_all_years", rho_all, len(g.dropna()))]
                rhos += [rho] if pop == "M16" else []
    return rows, {"rho": float(np.median(rhos))}, pd.DataFrame(aux)


def cause_g(df: pd.DataFrame) -> tuple:
    rows, lams, dqs, aux = [], [], [], []
    ch = pd.read_csv(P.REP / "P17_CHOICE.csv", comment="#")
    for m in D.MODS:
        a = df[(df["group"] == "M16") & (df["date"].dt.year >= 2017)].groupby(df["date"].dt.year)[[m + "_lam", m + "_c"]].median()
        lams += list(a[m + "_lam"])
        rows += [D.row("g", "M16", "years2017+", m + "_A1", "share_years_lambda_lt_1", float((a[m + "_lam"] < 1).mean()), len(a)), D.row("g", "M16", "years2017+", m + "_A1", "median_lambda", float(a[m + "_lam"].median()), len(a)),
                 D.row("g", "M16", "years2017+", m + "_A1", "median_c", float(a[m + "_c"].median()), len(a))]
        aux += [{"module": m, "config": "A1", "year": int(y), "lam": r[m + "_lam"], "c": r[m + "_c"]} for y, r in a.iterrows()]
        for cfg in ("B1", "B2"):
            k = ch[(ch["module"] == m) & (ch["config"] == cfg) & (ch["year"] >= 2017)]["kappa"]
            rows.append(D.row("g", "17", "years2017+", "%s_%s" % (m, cfg), "median_kappa", float(k.median()), len(k)))
            aux += [{"module": m, "config": cfg, "year": int(y), "kappa": v} for y, v in zip(ch[(ch["module"] == m) & (ch["config"] == cfg) & (ch["year"] >= 2017)]["year"], k)]
        for pop in D.POPS:
            for w in D.WINDOWS:
                d = D.sub(df, pop, w, ["k1", m + "_A1"])
                if d.empty:
                    continue
                f1 = E.apply_corr(d["k1"], d[m + "_g"], 0.0, d[m + "_c"].fillna(1.0).to_numpy(), 1.0, "qlike")
                q1, qa, qk = _q(d["rv"], f1).mean(), _q(d["rv"], d[m + "_A1"]).mean(), _q(d["rv"], d["k1"]).mean()
                rows += [D.row("g", pop, w, m + "_A1", "QLIKE_lambda1", float(q1), len(d)), D.row("g", pop, w, m + "_A1", "QLIKE_chosen_lambda", float(qa), len(d)), D.row("g", pop, w, m + "_A1", "QLIKE_K1", float(qk), len(d)),
                         D.row("g", pop, w, m + "_A1", "lambda1_minus_K1", float(q1 - qk), len(d))]
                dqs += [q1 - qa] if pop == "M16" and w == "all" else []
    return rows, {"lam_med": float(np.median(lams)), "dq_lam1": float(np.mean(dqs))}, pd.DataFrame(aux)


def cause_h(df: pd.DataFrame) -> tuple:
    rows, met = [], {}
    for pop in D.POPS:
        d = D.sub(df, pop, "all", ["k1", "lgbm_D2"])
        gp = d["gap"].astype(bool)
        dq = _q(d["rv"], d["lgbm_D2"]) - _q(d["rv"], d["k1"])
        a, b = float(dq.mean()), float(dq[~gp.to_numpy()].mean())
        rel = 9.99 if np.sign(a) != np.sign(b) else abs(b - a) / abs(a)
        rows += [D.row("h", pop, "all", "lgbm_D2", "share_gap_pairs", float(gp.mean()), len(d)), D.row("h", pop, "all", "lgbm_D2", "delta_D2_minus_K1_with_gap", a, len(d)),
                 D.row("h", pop, "all", "lgbm_D2", "delta_D2_minus_K1_without_gap", b, int((~gp).sum())), D.row("h", pop, "all", "lgbm_D2", "relative_change_without_gap", rel, len(d))]
        if pop == "M16":
            met = {"gap_share": float(gp.mean()), "rel_gap": rel}
    cal = pd.read_csv(P.REP / "P0_CALENDAR.csv", comment="#", parse_dates=["date"])
    c = cal[cal["date"] >= pd.Timestamp(P.FIRST_TARGET)][P.MAIN17]
    rows += [D.row("h", "17", "all", "calendar", "mean_share_of_17_absent_per_date", float(1 - c.to_numpy().mean()), len(c)), D.row("h", "17", "all", "calendar", "share_dates_with_any_absent", float((c.min(axis=1) == 0).mean()), len(c))]
    return rows, met, None
