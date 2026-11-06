"""E5 (G11): итоговые документы — FINAL (таблицы п. 21 и п. 33 Ж-206, раздел «Гибриды» с таблицей п. 11 Ж-210), HYBRIDS.md, REPRODUCTION.md, FINAL_CONFIG.json (XGB, LGBM),
STATISTICAL_TESTS.csv по каталогам, CHECKPOINT. Числа — только из рядов и таблиц E3/E4; выводы — из `CONCLUSIONS_E.md` (текст автора, числа в нём сверяются руками)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, docs_e, pe, ser, st_seeds
from models_v2.s6.xgb import c_calc, c_space, worldx

REP = pe.REP
FINAL = REP / "XGBOOST_LIGHTGBM_FULL_IMPROVEMENT_FINAL.md"
STATUS = {"hist": "HISTORICAL WALK-FORWARD (2017–02.2026, просмотрено; не независимый OOS); INDEPENDENT FUTURE TEST не выполнялся, нет данных после 27.02.2026"}


def md(df: pd.DataFrame, fmt: str = "%.4f") -> str:
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(fmt % v if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def row(df, n, label, tests, verdict, ref="HAR") -> dict:
    d = crit_e.describe(n, df[n].to_numpy(float), df)
    t = tests[(tests["candidate"] == n) & (tests["reference"] == ref) & (tests["period"] == "вся история")]
    return {"Модель": label, "id": n, "QLIKE": d["qlike_all"], "RMSE": d["rmse_lvl"], "MAE": d["mae_lvl"], "R²(MZ)": d["r2_mz_lvl"], "2017–2021": d["qlike_2017_2021"], "2022–2025": d["qlike_2022_2025"],
            "DM p (к %s)" % ref: float(t["p_raw"].iloc[0]) if len(t) else float("nan"), "p Холм": float(t["p_holm"].iloc[0]) if len(t) else float("nan"), "Статус": verdict}


def table33(df, tests, ar, fin) -> pd.DataFrame:
    bst = lambda a: next(n for n, v in fin.items() if v[0] == "A" and v[1] == a)  # noqa: E731
    v = lambda n: ar.set_index("id")["verdict"].get(n, "ссылка (baseline)")  # noqa: E731
    spec = [("HAR", "HAR-J baseline"), ("B2X", "XGBoost baseline (Б-2)"), ("XJ", "XGBoost improved (x2_best_j; 2017–2021 — потолок)"), ("B2L", "LightGBM baseline (аналог Б-2)"), ("LF", "LightGBM improved (C0_inh_F4)"),
            (bst("A3"), "XGBoost + LightGBM"), (bst("A1"), "HAR + XGBoost"), (bst("A2"), "HAR + LightGBM"), (bst("A4"), "HAR + XGBoost + LightGBM")]
    return pd.DataFrame([row(df, n, lab, tests, v(n)) for n, lab in spec])


def table11(df, tests, ar, fin) -> pd.DataFrame:
    pick = lambda k, f: next(n for n, x in fin.items() if x[0] == k and x[1] == f)  # noqa: E731
    spec = [("HAR", "HAR-J"), ("XJ", "XGBoost"), ("LF", "LightGBM"), (pick("A", "A1"), "HAR + XGBoost"), (pick("A", "A2"), "HAR + LightGBM"), (pick("A", "A3"), "XGBoost + LightGBM"),
            (pick("A", "A4"), "HAR + XGB + LGBM"), (pick("B", "1"), "HAR residual XGBoost"), (pick("B", "2"), "HAR residual LightGBM"), (next(n for n, x in fin.items() if x[0] == "C"), "Adaptive Hybrid")]
    t = pd.DataFrame([row(df, n, lab, tests, ar.set_index("id")["verdict"].get(n, "—")) for n, lab in spec])
    return t.drop(columns=["MAE", "R²(MZ)"])


def final_config(alg: str, w: dict) -> dict:
    sd, x2 = pe.read(pe.ENS / "SEEDS_E.csv"), comp.x2_moves()
    base = {"status": "DEVELOPMENT + " + STATUS["hist"], "expanding window, daily retrain": True, "features": "Б-2 (28) + F4 (5) = 33", "seeds": list(comp.SEEDS), "date": pd.Timestamp.now().strftime("%Y-%m-%d")}
    if alg == "xgboost":
        hp = c_space.apply(c_calc.hpc_base(w), x2[1])
        return {**base, "model": "x2_best_j", "trial_id": x2[0], "objective": "reg:squarederror, цель ln RV_{t+1} − ln RV_m; σ̂² = exp(off + f + s²/2)", "moves_vs_I2": x2[1], "hp_by_year": {str(Y): h.__dict__ for Y, h in hp.items()},
                "note": "ПОТОЛОК: оси подобраны по LOYO-годам 2017–2021 (кандидат видел их); чистая справочная — ctrl_f4", "control": "ctrl_f4", "seeds_qlike": sd[sd["id"] == "XJ"].iloc[0].filter(like="qlike_s").to_dict()}
    from models_v2.s6.lgbm import grid as lgrid
    return {**base, "model": "C0_inh_F4", "objective": "regression L2, ln-цель; наследованные HP годового выбора И2 (из сетки 132 XGBoost)", "hp_by_year": {str(Y): lgrid.from_i2(w["i2"][Y]).label() for Y in paths6.YEARS},
            "seeds_qlike": sd[sd["id"] == "LF"].iloc[0].filter(like="qlike_s").to_dict()}


def stage(_a=None) -> int:
    df, cdf = ser.load(), comp.load()
    tests, ar, sel = pe.read(pe.ENS / "STATISTICAL_TESTS_all.csv"), pe.read(pe.ENS / "ACCEPT_REJECT.csv"), pe.read(pe.ENS / "HYBRID_SELECTION.csv")
    fin, sd = st_seeds.finalists(df), pe.read(pe.ENS / "SEEDS_E.csv")
    t33, t11 = table33(df, tests, ar, fin), table11(df, tests, ar, fin)
    w = worldx.load()
    cfg = {a: final_config(a, w) for a in ("xgboost", "lightgbm")}
    for a, c in cfg.items():
        paths6.dump_json(c, REP / a / "FINAL_CONFIG.json")
    for d in ("xgboost", "lightgbm", "xgb_lgbm_ensemble", "har_xgboost", "har_lightgbm", "har_xgb_lgbm"):
        keep = {"xgboost": ("XJ", "XF"), "lightgbm": ("LF",), "xgb_lgbm_ensemble": tuple(n for n, x in fin.items() if x[:2] == ("A", "A3")) + ("XJ",), "har_xgboost": tuple(n for n in fin if n[:2] in ("A1", "B1")),
                "har_lightgbm": tuple(n for n in fin if n[:2] in ("A2", "B2")), "har_xgb_lgbm": tuple(n for n in fin if n[:2] in ("A4", "C2", "C3", "C4", "C5", "C6"))}[d]
        sub = tests[tests["candidate"].isin(keep)]
        pe.write(sub, REP / d / "STATISTICAL_TESTS.csv", "E3 (%s): DM, бутстрап, Холм (N), периоды" % d)
        if d != "xgb_lgbm_ensemble":
            shutil.copy(pe.CHECK, REP / d / "CHECKPOINT_E.json")
    pe.write(tests, REP / "STATISTICAL_TESTS.csv", "E3: все сравнения волны E")
    pe.write(sel, REP / "HYBRID_SELECTION.csv", "R9: выбор гибридов")
    pd.DataFrame(docs_e.table21(df, tests, sd)).to_csv(pe.ENS / "TABLE21.csv", index=False)
    (REP / "HYBRIDS.md").write_text(docs_e.hybrids_md(sel, ar, md(t11), fin, df))
    (REP / "REPRODUCTION.md").write_text(docs_e.repro_md())
    conc = (REP / "CONCLUSIONS_E.md").read_text() if (REP / "CONCLUSIONS_E.md").exists() else "(выводы автора: CONCLUSIONS_E.md не найден)\n"
    t21 = pd.read_csv(pe.ENS / "TABLE21.csv") if (pe.ENS / "TABLE21.csv").exists() else pd.DataFrame()
    FINAL.write_text("# XGBoost и LightGBM: полная доработка, SBER h=1 — итог волны E\n\nСтатус: %s. Первая цифра — потолок. Все p — после выбора на тех же данных.\n\n## Таблица п. 21 (XGBoost против LightGBM)\n\n%s\n## Таблица п. 33\n\n%s\n"
                     "## Гибриды (таблица п. 11 Ж-210)\n\n%s\nACCEPT/REJECT — `xgb_lgbm_ensemble/ACCEPT_REJECT.csv`, выбор A–F — `HYBRID_SELECTION.csv`. Подробности — `HYBRIDS.md`.\n\n%s\n" %
                     (STATUS["hist"], md(t21) if len(t21) else "(TABLE21.csv отсутствует)", md(t33), md(t11), conc))
    paths6.dump_json({"n_trials": int(pe.n_trials()), "t11": t11["QLIKE"].round(6).tolist()}, pe.ENS / "E5_SUMMARY.json")
    print("G11 e5: FINAL записан;", "таблица п. 33 — %d строк, п. 11 — %d строк" % (len(t33), len(t11)), flush=True)
    return 0
