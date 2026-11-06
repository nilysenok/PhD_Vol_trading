"""Ж-212 (HS10, HS11, G13): стадия `g` — единая проверка 10 моделей п. 9 блока на 2271 дате: метрики по периодам/годам/режимам, крупнейшие ошибки, корреляции, чувствительность, тесты A/B/C
(DM Ньюи–Уэст лаг 5, блочный бутстрап 20/5000/seed 42), Холм с N = 521 + trials Ж-212 (объявлено в CHECKPOINT.json ДО расчёта p), MCS, уровни 1–3 (уровень 4 — «не проверялся»).
Финалисты направлений — минимум crit_wf по всей истории (потолок; метка генерируется кодом)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import stats
from models_v2.s5.d import stats_d
from models_v2.s6.e import crit_e
from models_v2.s7 import hf_levels as V
from models_v2.s8 import hs_eval, hs_levels as LV, hs_ledger as L, hs_paths as P, hs_store

PERIODS = {"2017_2021": (2017, 2021), "2019_2021": (2019, 2021), "2022": (2022, 2022), "2023_2025": (2023, 2025), "2026": (2026, 2026), "all": (2017, 2026)}
FILES = (("B", "HAR_FEATURES_B.csv", "B", "family"), ("C", "RESIDUAL_QLIKE_C.csv", "C", "family"))


def best_row(csv: str, fam: str | None = None):
    p = P.REP / csv
    if not p.exists():
        return None
    t = pd.read_csv(p, comment="#")
    if "crit_wf" not in t.columns:
        return None
    if fam is not None:
        t = t[t["family"] == fam]
    return None if t.empty else t.sort_values("crit_wf").iloc[0]["model"]


def collect() -> dict:
    """-> {имя модели: (σ̂², хранилище, ключ)}; отсутствующие направления пропускаются (записывается в «не запускалось»)."""
    fro = hs_store.frozen()
    out = {"HAR-J": fro["HAR"].to_numpy(float), "XGBoost": fro["XJ"].to_numpy(float), "LightGBM": fro["LF"].to_numpy(float)}
    for lab, csv, store, fam in (("B-XGB", "HAR_FEATURES_B.csv", "B", "XGB"), ("B-LGBM", "HAR_FEATURES_B.csv", "B", "LGBM"), ("C-XGB", "RESIDUAL_QLIKE_C.csv", "C", "XGB"),
                                 ("C-LGBM", "RESIDUAL_QLIKE_C.csv", "C", "LGBM"), ("D", "COMPONENT_D.csv", "D", None), ("E", "CONDITIONAL_E.csv", "E", None), ("F", "MOE_F.csv", "F", None)):
        nm = best_row(csv, fam)
        if nm is not None:
            out["%s:%s" % (lab, nm)] = hs_store.get(store, nm)
    return out


def metric_rows(df, S: dict) -> pd.DataFrame:
    rows = []
    for nm, s2 in S.items():
        for pk, (lo, hi) in PERIODS.items():
            m = hs_eval.period_mask(df, lo, hi)
            fm = stats.full_metrics(df["rv"].to_numpy(float)[m], s2[m])
            q = np.asarray(metrics.qlike(df["rv"].to_numpy(float)[m], s2[m]), float)
            rows.append({"model": nm, "period": pk, "n": int(m.sum()), "qlike": float(q.mean()), **{k: fm[k] for k in ("rmse_lvl", "mae_lvl", "r2_mz_lvl", "rmse_ln", "mae_ln", "bias_ln", "mean_r")}})
    m = V.mask_of(df, "no_240222")
    for nm, s2 in S.items():
        rows.append({"model": nm, "period": "no_240222", "n": int(m.sum()), "qlike": float(np.asarray(metrics.qlike(df["rv"].to_numpy(float)[m], s2[m])).mean())})
    return pd.DataFrame(rows)


def annual_rows(df, S: dict) -> pd.DataFrame:
    y, rv = hs_eval.years_of(df), df["rv"].to_numpy(float)
    Q = {k: np.asarray(metrics.qlike(rv, v), float) for k, v in S.items()}
    return pd.DataFrame([{"model": k, "year": yy, "qlike": float(Q[k][y == yy].mean()), "d_vs_HAR": float((Q[k] - Q["HAR-J"])[y == yy].mean()),
                          "d_vs_XGB": float((Q[k] - Q["XGBoost"])[y == yy].mean()), "d_vs_LGBM": float((Q[k] - Q["LightGBM"])[y == yy].mean())} for k in S for yy in hs_eval.YEARS])


def regime_rows(df, S: dict) -> pd.DataFrame:
    o = pd.read_parquet(P.OUT / "ERROR_STRUCTURE_DAILY.parquet")
    if not (o["d"].to_numpy() == df["d"].to_numpy()).all():
        raise SystemExit("стоп: сетка таблицы A != сетка компонентов")
    rv, rows = df["rv"].to_numpy(float), []
    for nm, s2 in S.items():
        q = np.asarray(metrics.qlike(rv, s2), float)
        for rg, col in (("ter_rv_w_SBER", "ter_rv_w"), ("ter_mkt_IMOEX", "mkt_ter_w"), ("RVI", "rvi_regime"), ("stress_G4", "st_g4")):
            for v in sorted(pd.unique(o[col]), key=str):
                m = (o[col] == v).to_numpy()
                rows.append({"model": nm, "regime": rg, "value": str(v), "n": int(m.sum()), "qlike": float(q[m].mean())})
    return pd.DataFrame(rows)


def errors_and_corr(df, S: dict) -> tuple:
    rv = df["rv"].to_numpy(float)
    E = pd.DataFrame({k: np.log(rv / v) for k, v in S.items()})
    top = []
    for k, v in S.items():
        q = np.asarray(metrics.qlike(rv, v), float)
        for i in np.argsort(-q)[:10]:
            top.append({"model": k, "d": str(df["d"].iloc[i].date()), "target_date": str(df["target_date"].iloc[i].date()), "rv": rv[i], "sigma2": v[i], "qlike": q[i]})
    return pd.DataFrame(top), E.corr().reset_index().rename(columns={"index": "model"})


def sensitivity(S_names: list) -> pd.DataFrame:
    """Плато: размах crit_wf среди конфигураций той же модели/направления и число соседей в пределах 0,0005 от финалиста."""
    rows = []
    for csv, key in (("HAR_FEATURES_B.csv", "family"), ("RESIDUAL_QLIKE_C.csv", "family"), ("COMPONENT_D.csv", None), ("CONDITIONAL_E.csv", None), ("MOE_F.csv", None)):
        p = P.REP / csv
        if not p.exists():
            continue
        t = pd.read_csv(p, comment="#")
        if "crit_wf" not in t.columns:
            continue
        for fam, g in (t.groupby(key) if key else [("all", t)]):
            b = g.sort_values("crit_wf").iloc[0]
            rows.append({"direction": csv.split(".")[0], "group": fam, "best": b["model"], "crit_best": b["crit_wf"], "crit_min": g["crit_wf"].min(), "crit_max": g["crit_wf"].max(),
                         "n_configs": len(g), "n_within_0.0005": int((g["crit_wf"] - b["crit_wf"] <= 0.0005).sum()), "plateau": bool(len(g) > 1 and (g["crit_wf"] - b["crit_wf"] <= 0.0005).sum() >= 2)})
    return pd.DataFrame(rows)


def cost_rows() -> pd.DataFrame:
    rows = []
    for csv in ("HAR_FEATURES_B.csv", "RESIDUAL_QLIKE_C.csv", "COMPONENT_D.csv", "CONDITIONAL_E.csv", "MOE_F.csv"):
        p = P.REP / csv
        if p.exists():
            t = pd.read_csv(p, comment="#")
            rows.append({"direction": csv.split(".")[0], "n_rows": len(t), "secs_total": float(t["secs"].sum()) if "secs" in t and t["secs"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def tests_and_levels(df, S: dict, N: int) -> tuple:
    rows = []
    for k in S:
        if k not in LV.REFS:
            rows += V.tests_rows(df, k, S[k], N)
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    ix = t.index[(t["period"] == "all") & t["test"].isin(["A", "B"])]
    t.loc[ix, "p_holm"] = stats_d.holm_n(t.loc[ix, "p_raw"].fillna(1.0).to_numpy(), N)
    t["post_selection"] = "постселекционно: финалист выбран по crit_wf на всей истории; p не поправлены на этот выбор; HISTORICAL WALK-FORWARD, не OOS"
    q = {k: np.asarray(metrics.qlike(df["rv"].to_numpy(float), v), float) for k, v in S.items()}
    cw = {k: crit_e.cw(v, df["target_date"]) for k, v in q.items()}
    crit, qall = {k: cw[k] for k in S}, {k: float(q[k].mean()) for k in S}
    crit.update({"HAR": cw["HAR-J"], "S*": cw["XGBoost"]})
    qall["HAR"] = qall["HAR-J"]
    lv = [{"model": k, "crit_wf": cw[k], "qlike_all": qall[k], **LV.levels(t, k, crit, qall, q[k] - q["XGBoost"])} for k in S if k not in LV.REFS]
    return t, pd.DataFrame(lv)


def comparisons_note(S: dict) -> str:
    """Шапка FORECAST_COMPARISONS: фактическое число моделей и не запускавшиеся направления."""
    miss = "; E не запускалась (HS8/HS11)" if not any(k.startswith("E:") for k in S) else ""
    return "G: %d моделей п. 9 на единых 2271 датах%s; HISTORICAL WALK-FORWARD, не OOS; финалисты направлений — потолок (выбор по crit_wf на всей истории)" % (len(S), miss)


def write_comparisons(_a=None) -> int:
    df, S = hs_store.frozen(), collect()
    P.write_csv(metric_rows(df, S), "FORECAST_COMPARISONS.csv", comparisons_note(S))
    return 0


def stage_g(_a=None) -> int:
    N = L.n_final()
    L.write_check(L.load(), N_declared_before_stage_g=N, N_for_corrections=N)
    df, S = hs_store.frozen(), collect()
    P.write_csv(metric_rows(df, S), "FORECAST_COMPARISONS.csv", comparisons_note(S))
    P.write_csv(annual_rows(df, S), "ANNUAL_PERFORMANCE.csv", "G: QLIKE по годам и Δ к HAR-J / XGB / LGBM (минус — модель лучше)")
    P.write_csv(regime_rows(df, S), "REGIME_PERFORMANCE.csv", "G: QLIKE по режимам (терцили RV_w SBER и IMOEX — по прошлому, RVI, стресс Г4)")
    top, corr = errors_and_corr(df, S)
    P.write_csv(top, "LARGEST_ERRORS.csv", "G: 10 крупнейших по QLIKE дней каждой модели")
    P.write_csv(corr, "ERROR_CORRELATION.csv", "G: корреляция ln-ошибок ln(RV/σ̂²) между моделями")
    P.write_csv(sensitivity(list(S)), "SENSITIVITY.csv", "G: чувствительность — размах crit_wf среди конфигураций направления; плато: >= 2 конфигураций в пределах 0,0005 от лучшей")
    P.write_csv(cost_rows(), "COMPUTE_COST.csv", "G: вычислительная стоимость (секунды счёта строк; D/E/F без измерения — NaN)")
    t, lv = tests_and_levels(df, S, N)
    P.write_csv(t, "STATISTICAL_TESTS.csv", "G: A (к HAR-J), B (к S*=XJ), C (к LF): Δ=QLIKE(M)−QLIKE(опоры), DM НW лаг 5 двусторонний, бутстрап блок 20 / 5000 / seed 42; Холм (тесты A и B, период all) с N=%d" % N)
    P.write_csv(lv, "LEVELS.csv", "G: уровни 1–3 по HS11; уровень 4 — не проверялся; ACCEPTED не присваивается")
    Lm = np.column_stack([np.asarray(metrics.qlike(df["rv"].to_numpy(float), v), float) for v in S.values()])
    mc = stats_d.mcs(Lm, stats_d.block_idx(len(Lm)))
    P.write_csv(pd.DataFrame({"model": list(S), "p_mcs": mc["p"], "in_mcs_90": mc["in"]}), "MCS.csv", "G: Model Confidence Set 90 % (бутстрап блок 20, 5000, seed 42) по QLIKE %d моделей" % len(S))
    print(lv[["model", "qlike_all", "crit_wf", "L1_a", "L1_b", "L1_c", "L1_d", "L2_b", "L2_c", "L3", "verdict"]].round(5).to_string(index=False), flush=True)
    return 0
