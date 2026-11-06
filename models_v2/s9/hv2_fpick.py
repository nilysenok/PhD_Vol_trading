"""Ж-214, лист 5 (V12, У30): финалисты ступеней Л1–Л9 и лучшей V8-строки — механически, минимум `crit_wf` среди кандидатов ступени (выбор по всей истории — потолок).
Z6 (другая цель), J3 (не прогноз RV), H7/H8/H9 в Л2 (QLIKE-подгонка), H10/H11 (плато-соседи H2), диагностики «|lam1» в семейство не входят (EXCLUDED)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_ledger as L, hv2_paths as P

H, HR = "harq_j/HARQ_J.csv", "harq_j/HARQ_J_REST.csv"
RULES = (
    ("Л1", "QLIKE-HAR-J", (("harj_qlike/QLIKE_HARJ.csv", ("Q1", "Q2", "Q4", "Q5")),)),
    ("Л2", "HARQ-J (без QLIKE)", ((H, ("H1", "H4")), (HR, ("H2", "H3", "H5", "H6")))),
    ("Л3", "QLIKE-HARQ-J", ((H, ("H7",)), (HR, ("H8",)))),
    ("Л4", "HAR-J + Intraday (схема T)", (("intraday_decomposition/DECOMP_T.csv", ("T1", "T3")), ("intraday_decomposition/DECOMP_T_REST.csv", ("T2", "T4", "P1", "P2")))),
    ("Л5", "HAR-J + Jump/Semivariance (схемы S, J)", (("semivariance_jumps/DECOMP_S.csv", ("S1", "S3")), ("semivariance_jumps/DECOMP_J.csv", ("J1",)))),
    ("Л6", "HAR-J + Structural XGB", (("xgboost_structural/STRUCT_XGB.csv", ("N1x_A0",)),)),
    ("Л7", "HAR-J + Structural LGBM", (("lightgbm_structural/STRUCT_LGBM.csv", ("N1l_A0",)),)),
    ("Л8", "HAR-J + Market commonality", (("market_commonality/MARKET_COMMONALITY.csv", tuple("E%d" % i for i in range(1, 10))),)),
    ("Л9", "HAR-J + Reconciliation", (("reconciliation/RECON.csv", ("R2T",)),)),
    ("V8", "HAR-J, сезонность (вход, input_measurement)", (("seasonality/SEASONALITY_V8.csv", ("Z1", "Z2", "Z4", "Z5")), ("seasonality/SEASONALITY_V8_REST.csv", ("Z3", "Z7")))),
)
EXCLUDED = "Z6 (target_definition, справочно); J3 (прогноз скачка, не RV); H9 (QLIKE-подгонка, У9) в Л2; H10, H11 (плато H2); строки «|lam1» (диагностика)"
CONTROLS = {"A0", "XJ", "LF", "D0_drop_a", "bestQ", "Ca_x", "Cc_x", "Ca_l", "Cc_l", "A1_roll_126", "Bstar_H2"}
SERIES = ("Q", "H", "H2", "DEC", "DEC2", "SEA", "SEA2", "ML", "MKT", "REC")


def read_table(rel: str) -> pd.DataFrame:
    return pd.read_csv(P.REP / rel, comment="#")


def choose(crit: dict, order: tuple) -> str:
    """Правило V12: минимум crit_wf среди кандидатов; ничья — раньше в списке. Не по QLIKE всей истории и не по какому-либо другому столбцу."""
    missing = [n for n in order if n not in crit or not np.isfinite(crit[n])]
    if missing:
        raise KeyError("нет crit_wf у кандидатов %s" % missing)
    return min(order, key=lambda n: (crit[n], order.index(n)))


def finalists() -> pd.DataFrame:
    rows = []
    for step, label, files in RULES:
        crit, qall, order = {}, {}, tuple(n for _, ns in files for n in ns)
        for rel, names in files:
            t = read_table(rel).set_index("model")
            for n in names:
                crit[n], qall[n] = float(t.loc[n, "crit_wf"]), float(t.loc[n, "qlike_all"])
        m = choose(crit, order)
        rows.append({"step": step, "label": label, "model": m, "crit_wf": crit[m], "qlike_all": qall[m], "n_candidates": len(order), "candidates": ",".join(order),
                     "crit_best_other": min(crit[n] for n in order if n != m) if len(order) > 1 else np.nan, "excluded": EXCLUDED})
    return pd.DataFrame(rows)


def load_series(names, grid: pd.DataFrame) -> dict:
    """Ряды σ̂² финалистов из `<направление>_series.parquet`; сетка (d, target_date, rv) обязана совпасть с замороженной."""
    out = {}
    for s in SERIES:
        t = pd.read_parquet(P.OUT / ("%s_series.parquet" % s))
        for n in names:
            if n in t.columns and n not in out:
                if not ((t["d"].to_numpy() == grid["d"].to_numpy()).all() and (t["target_date"].to_numpy() == grid["target_date"].to_numpy()).all()
                        and np.array_equal(t["rv"].to_numpy(float), grid["rv"].to_numpy(float))):
                    raise SystemExit("стоп: сетка %s не совпадает с замороженной" % s)
                out[n] = t[n].to_numpy(float)
    miss = [n for n in names if n not in out]
    if miss:
        raise KeyError("нет рядов: %s" % miss)
    return out


def registry_spec(name: str) -> dict:
    """Спецификация конфигурации из реестра Ж-214 (поле name в JSON конфигурации); нет — пусто."""
    df = L.load()
    for _, r in df[df["program"] == L.PROGRAM].iterrows():
        c = json.loads(r["config"])
        if c.get("name") == name:
            return {"key": r["key"], "direction": r["direction"], **c}
    return {}


def robust_rows(skip: set) -> pd.DataFrame:
    """Справочно, вне семейства: строки всех направлений, устойчивые без 24.02.2022 (ΔQLIKE к A0 < 0 на всей истории и без 24.02.2022, лет лучше >= 6); выбраны после просмотра."""
    keep, rows = {n for _, _, f in RULES for _, ns in f for n in ns}, []
    for rel in sorted({rel for _, _, f in RULES for rel, _ in f}):
        t = read_table(rel)
        t = t[~t["model"].isin(CONTROLS) & ~t["model"].str.contains(r"\|", regex=True) & t["d_qlike_A0"].notna()]
        if "STRUCT" in rel:
            t = t[t["model"].isin(keep)]
        rows.append(t.assign(file=rel))
    a = pd.concat(rows, ignore_index=True).drop_duplicates("model")
    ok = (a["d_qlike_A0"] < 0) & (a["d_qlike_no_240222_A0"] < 0) & (a["years_better_A0"] >= 6)
    a = a[ok & ~a["model"].isin(skip)]
    return a[["model", "file", "crit_wf", "qlike_all", "d_crit_A0", "d_qlike_A0", "d_qlike_no_240222_A0", "years_better_A0"]].sort_values("d_qlike_no_240222_A0").reset_index(drop=True)
