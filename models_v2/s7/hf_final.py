"""Стадия final (GATES_HF G9, HF9–HF11): сводка H1–H7, срезы, тесты A/B/C с Холмом по N, уровни 1/2, таблица п. 17 SBER h=1, документы.
Финалисты внутри семейства — min crit_wf по ряду кандидатов (потолок: фиксированные c/λ выбраны на всей истории), отдельно — лучший по причинному правилу `*_wf`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, ser
from models_v2.s7 import hf_bound as B, hf_cmp as C, hf_data as D, hf_docs, hf_feat as FE, hf_ledger as L, hf_levels as V, hf_wstage as W

FAM_NOTE = {"H1": "HAR+XGB (комбинация)", "H2": "HAR+LGBM (комбинация)", "H3": "Residual XGBoost", "H4": "Residual LightGBM", "H5": "Adaptive Hybrid (режимы отклонены → C6/C5 Ж-210)",
            "H7": "HAR+XGB+LGBM"}


def candidates(df: pd.DataFrame) -> dict:
    """{имя: (σ̂², семейство, wf?, перенос?)} по сохранённым рядам (новые — стадии bound/weights/feat; перенос — series_a / series_c Ж-210)."""
    out, sa, sc = {}, pd.read_parquet(ser.file("a")), pd.read_parquet(ser.file("c"))
    for k, f in (("A1_roll_126", "H1"), ("A2_roll_252", "H2"), ("A4_roll_252", "H7")):
        out[k] = (sa[k].to_numpy(float), f, False, True)
    for k in ("C6", "C5"):
        out[k] = (sc[k].to_numpy(float), "H5", True, True)
    for f in (W.SERIES, B.SERIES, FE.SERIES):
        s = pd.read_parquet(f)
        for k in s.columns[3:]:
            fam = k[:2] if k[:2] in ("H1", "H2", "H7") else ("H3" if "xgb" in k else "H4")
            out[k] = (s[k].to_numpy(float), fam, "wf" in k and not k.startswith("R1"), k.startswith("R1_"))
    return out


def finalists(df, cand) -> dict:
    """{метка: имя ряда}: лучший по crit_wf в семействе; для H3/H4 дополнительно лучший причинный (wf)."""
    crit = {k: C.crit_e.cw(D.qs(df, v[0]), df["target_date"]) for k, v in cand.items()}
    fin = {}
    for fam in ("H1", "H2", "H3", "H4", "H5", "H7"):
        names = [k for k, v in cand.items() if v[1] == fam]
        fin[fam] = min(names, key=lambda k: (crit[k], k))
        wf = [k for k in names if cand[k][2]]
        if fam in ("H3", "H4") and wf:
            fin[fam + "_wf"] = min(wf, key=lambda k: (crit[k], k))
    return fin


def trunc_summary() -> dict:
    mx = {}
    for f in ("RESIDUAL_XGBOOST.csv", "RESIDUAL_LIGHTGBM.csv", "DYNAMIC_WEIGHTS.csv", "FEATURE_SELECTION.csv"):
        t = pd.read_csv(L.REP / f, comment="#")
        for c in ("trunc_max_diff", "leak_max_diff"):
            if c in t:
                mx[f + ":" + c] = float(t[c].max()) if t[c].notna().any() else None
    return mx


def comparisons(df, series: dict) -> tuple:
    names = list(series)
    q = {k: D.qs(df, v) for k, v in series.items()}
    rows, ann = [], []
    for key in V.SLICES:
        m = V.mask_of(df, key)
        qs_ = {k: float(q[k][m].mean()) for k in names}
        rk = pd.Series(qs_).rank()
        rows += [{"period": key, "model": k, "qlike": qs_[k], "rank": int(rk[k])} for k in names]
    y = D.year(df)
    for k in names:
        for yy in range(2017, 2027):
            m = y == yy
            ann.append({"model": k, "year": yy, "qlike": float(q[k][m].mean()), "d_vs_HAR": float((q[k] - q["HAR-J"])[m].mean()), "d_vs_XGB": float((q[k] - q["XGBoost"])[m].mean()),
                        "d_vs_LGBM": float((q[k] - q["LightGBM"])[m].mean())})
    return pd.DataFrame(rows), pd.DataFrame(ann)


def robustness(df, series: dict) -> pd.DataFrame:
    """Диагностика (не для выбора): без лучшего года, без 5 лучших дней, спокойный/стресс (Г4), терцили RV; Δ к HAR-J и к XGBoost."""
    q = {k: D.qs(df, v) for k, v in series.items()}
    y, st = D.year(df), df["st_g4"].to_numpy(bool)
    ter = pd.read_csv(L.REP / "HAR_RESIDUAL_DIAGNOSTICS.csv", comment="#")["ter_rv_w"].to_numpy()
    rows = []
    for k in series:
        for ref in ("HAR-J", "XGBoost"):
            dl = q[k] - q[ref]
            yrs = {yy: float(dl[y != yy].mean()) for yy in range(2017, 2027)}
            top = np.argsort(dl)[:5]
            keep = np.ones(len(dl), bool)
            keep[top] = False
            rows.append({"model": k, "ref": ref, "delta_all": float(dl.mean()), "delta_wo_best_year": max(yrs.values()), "delta_wo_top5_days": float(dl[keep].mean()),
                         "delta_calm": float(dl[~st].mean()), "delta_stress": float(dl[st].mean()), **{"delta_rv_ter%d" % t: float(dl[ter == t].mean()) for t in (0, 1, 2)}})
    return pd.DataFrame(rows)


def stage(_a=None) -> int:
    df, cand = comp.load(), None
    cand = candidates(df)
    fin = finalists(df, cand)
    n_final = L.n_final()
    series = {"HAR-J": comp.col(df, "HAR"), "XGBoost": comp.col(df, "XJ"), "LightGBM": comp.col(df, "LF")}
    labels = {}
    for lab, nm in fin.items():
        labels[lab + ": " + nm] = nm
        series[lab + ": " + nm] = cand[nm][0]
    cmp_, ann = comparisons(df, series)
    D.write(cmp_, "FORECAST_COMPARISONS.csv", "QLIKE и ранг по периодам HF10; финалисты — лучшие по crit_wf семейства (потолок)")
    D.write(ann, "ANNUAL_PERFORMANCE.csv", "QLIKE по годам и Δ к HAR-J/XGB/LGBM (минус — модель лучше)")
    D.write(robustness(df, series), "ROBUSTNESS.csv", "диагностика, не выбор: без лучшего года / без 5 лучших дней / спокойный–стресс / терцили RV SBER")
    rows = []
    for lab in [k for k in series if ":" in k]:
        rows += V.tests_rows(df, lab, series[lab], n_final)
    t = V.add_holm(pd.DataFrame(rows), n_final)
    t["post_selection"] = V.POSTSEL
    D.write(t, "STATISTICAL_TESTS.csv", "A/B/C: DM (НW лаг 5, двусторонний p) и блочный бутстрап (20, 5000, seed 42); Холм с N=%d по семейству (финалист × A/B/C, вся история); постселекционно" % n_final)
    sk, cs = C.star(df)
    crit = {"HAR": C.crit_e.cw(D.qs(df, series["HAR-J"]), df["target_date"]), sk: cs[sk]}
    qall = {"HAR": float(D.qs(df, series["HAR-J"]).mean())}
    lv = {}
    for lab in [k for k in series if ":" in k]:
        crit[lab], qall[lab] = C.crit_e.cw(D.qs(df, series[lab]), df["target_date"]), float(D.qs(df, series[lab]).mean())
        lv[lab] = V.levels(t, lab, crit, qall, sk)
    L1 = pd.DataFrame([{"model": k, **v, "ceiling_note": V.ceiling_note(k)} for k, v in lv.items()])
    D.write(L1, "LEVELS.csv", "HF11: уровень 1 / уровень 2 по каждому финалисту; star = %s" % sk)
    tr = trunc_summary()
    print("FINAL: N=%d; финалисты %s" % (n_final, fin), flush=True)
    print(L1[["model", "L1_a", "L1_b", "L1_c", "L1_d", "L2_b", "L2_c", "L2_d", "L2_e", "verdict"]].to_string(index=False), flush=True)
    print("FINAL: обрезка/порча", tr, flush=True)
    cfg = {"star": sk, "N_prior": L.counts(L.load())["N_prior"], "N_final": n_final, "finalists": fin, "levels": {k: v["verdict"] for k, v in lv.items()}, "truncation": tr}
    (L.REP / "FINAL_HYBRID_CONFIG.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    hf_docs.write_all(df, series, fin, t, L1, cmp_, ann, tr, n_final, sk)
    return 0
