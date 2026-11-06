"""Ж-214, лист 5 (V12, V15, У31, У32): абляционная лестница Л0–Л9 (`ABL_LADDER.csv`), итоговая таблица п. 29 блока, рейтинг архитектур по правилам (не по минимальной QLIKE).
Статус «потолок» и «HISTORICAL WALK-FORWARD, не OOS» генерируются кодом в каждой строке."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s8 import hs_eval
from models_v2.s9 import hv2_fpick as FP, hv2_fstats as FS

OWN = {"Л4": "T3", "Л8": "E1", "Л9": "T3"}      # ступень на ступень: Л4 = T3 + QLIKE-подгонка, Л8 против Mb (V10), Л9 против простой суммы T3 (V11); Л3 — лучшая из Л1/Л2
PER = (("2017_2021", 2017, 2021), ("2019_2021", 2019, 2021), ("2022", 2022, 2022), ("2023_2025", 2023, 2025), ("2026", 2026, 2026))
STATUS = "потолок; HISTORICAL WALK-FORWARD, не OOS"
RATING = ("c1_dcrit", "c2_years", "c3_ex_240222", "c4_sign_2of3", "c5_holm_A", "c6_wo_top5")


def own_base(step: str, fin: pd.DataFrame) -> str:
    if step == "Л3":
        c = fin[fin["step"].isin(["Л1", "Л2"])].sort_values("crit_wf").iloc[0]
        return c["model"]
    return OWN.get(step, "A0")


def why(r: dict, share22: float) -> str:
    """Только измеренные свойства: Δ на всей истории и без 24.02.2022, доля выигрыша на 2022, лет лучше; выводов о причинах из важности признаков нет."""
    return ("ΔQLIKE к A0 %.4f (без 24.02.2022 %.4f); доля суммарного выигрыша на 2022: %s; лет лучше A0: %d из 10; профиль коэффициентов — файлы направления"
            % (r["d_qlike"], r["d_qlike_no_240222"], "н/д" if not np.isfinite(share22) else "%.0f %%" % (100 * share22), r["years_better"]))


def ladder(df: pd.DataFrame, s: dict, fin: pd.DataFrame, extra: dict, top5: pd.DataFrame) -> pd.DataFrame:
    rv, y = df["rv"].to_numpy(float), hs_eval.years_of(df)
    ser = {"A0": s["HAR-J"], "XJ": s["XGBoost"], "LF": s["LightGBM"], **extra, **{f["model"]: s["%s:%s" % (f["step"], f["model"])] for _, f in fin.iterrows()}}
    q = {k: np.asarray(metrics.qlike(rv, v), float) for k, v in ser.items()}
    rows = [{"step": "Л0", "model": "A0", "spec": "HAR-J(б): OLS в логарифмах, d/w/m + ln(1+J), расширяющееся окно", "own_base": "-", "status": "контроль"}]
    for _, f in fin.iterrows():
        k, m = "%s:%s" % (f["step"], f["model"]), f["model"]
        sp, ob = FP.registry_spec(m), own_base(f["step"], fin)
        r = {"step": f["step"], "model": m, "label": f["label"], "spec": sp.get("spec", "н/д"), "key": sp.get("key", ""), "own_base": ob, "status": STATUS}
        for ref, a in (("A0", ser["A0"]), ("XJ", ser["XJ"]), ("LF", ser["LF"]), ("own", ser[ob] if ob != "A0" else ser["A0"])):
            c = hs_eval.row(df, m, s[k], a)
            r.update({"d_crit_%s" % ref: c["d_crit"], "d_qlike_%s" % ref: c["d_qlike"], "d_qlike_no_240222_%s" % ref: c["d_qlike_no_240222"], "years_better_%s" % ref: c["years_better"]})
            if ref == "A0":
                ca, r["crit_wf"], r["qlike_all"] = c, c["crit_wf"], c["qlike_all"]
        d = np.asarray(metrics.qlike(rv, s[k]), float) - q["A0"]
        for nm, lo, hi in PER:
            r["dq_A0_" + nm] = float(d[(y >= lo) & (y <= hi)].mean())
        t5 = top5[(top5["model"] == k) & (top5["ref"] == "A0")].iloc[0]
        r["dq_A0_wo_top5"] = t5["delta_wo_top5"]
        sh = float(d[y == 2022].sum() / d.sum()) if d.sum() < 0 else np.nan
        r["share_gain_2022"], r["why_measured"] = sh, why(ca, sh)
        rows.append(r)
    return pd.DataFrame(rows)


def rating(lad: pd.DataFrame, t: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """Шесть правил V14-устойчивости к A0; ранг: число выполненных, затем ΔQLIKE без 24.02.2022 (не минимальная QLIKE). Контроли и Л0 не ранжируются."""
    out = []
    for _, r in lad[lad["step"] != "Л0"].iterrows():
        a = t[(t["model"] == "%s:%s" % (r["step"], r["model"])) & (t["test"] == "A")].set_index("period")
        c = {"c1_dcrit": r["d_crit_A0"] <= -hs_eval.CRIT_STEP, "c2_years": r["years_better_A0"] >= hs_eval.MIN_YEARS, "c3_ex_240222": r["d_qlike_no_240222_A0"] < 0,
             "c4_sign_2of3": sum(r["dq_A0_" + p] < 0 for p in ("2019_2021", "2022", "2023_2025")) >= 2, "c5_holm_A": bool(a.loc["all", "p_holm"] < 0.10),
             "c6_wo_top5": r["dq_A0_wo_top5"] < 0}
        out.append({"step": r["step"], "model": r["model"], **{k: bool(v) for k, v in c.items()}, "passed": int(sum(c.values())), "d_qlike_no_240222_A0": r["d_qlike_no_240222_A0"], "d_crit_A0": r["d_crit_A0"],
                    "p_raw_A": a.loc["all", "p_raw"], "p_holm_A": a.loc["all", "p_holm"]})
    o = pd.DataFrame(out).sort_values(["passed", "d_qlike_no_240222_A0"], ascending=[False, True]).reset_index(drop=True)
    o["rank"] = np.arange(1, len(o) + 1)
    return o


def p29(met: pd.DataFrame, t: pd.DataFrame, lv: pd.DataFrame, fin: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Таблица п. 29 блока: одинаковые даты и n у всех строк, неопределённость Δ (ДИ 95 % блочного бутстрапа, DM-p сырой и Холма)."""
    lab = {"HAR-J": ("HAR-J", "A0"), "XGBoost": ("XGBoost (S*, контроль)", "XJ"), "LightGBM": ("LightGBM (контроль)", "LF")}
    order = ["HAR-J"] + ["%s:%s" % (r["step"], r["model"]) for _, r in fin.iterrows()] + ["XGBoost", "LightGBM"]
    for _, r in fin.iterrows():
        lab["%s:%s" % (r["step"], r["model"])] = ("%s [%s]" % (r["label"], r["model"]), r["model"])
    m, rows = met[met["period"] == "all"].set_index("model"), []
    for k in order:
        x = {"model": lab[k][0], "config": lab[k][1], "n": int(m.loc[k, "n"]), "date_first": str(df["target_date"].iloc[0].date()), "date_last": str(df["target_date"].iloc[-1].date()),
             "qlike": m.loc[k, "qlike"], "rmse": m.loc[k, "rmse_lvl"], "mae": m.loc[k, "mae_lvl"], "r2_mz": m.loc[k, "r2_mz_lvl"]}
        x["dq_vs_HARJ"], x["dq_vs_XGB"] = m.loc[k, "qlike"] - m.loc["HAR-J", "qlike"], m.loc[k, "qlike"] - m.loc["XGBoost", "qlike"]
        if k in FS.REFS:
            x["level"], x["status"] = "-", "контроль (заморожен), HISTORICAL WALK-FORWARD, не OOS"
        else:
            for tst, nm in (("A", "HARJ"), ("B", "XGB")):
                z = t[(t["model"] == k) & (t["test"] == tst) & (t["period"] == "all")].iloc[0]
                x.update({"boot_lo95_%s" % nm: z["boot_lo95"], "boot_hi95_%s" % nm: z["boot_hi95"], "p_dm_%s" % nm: z["p_raw"], "p_holm_%s" % nm: z["p_holm"]})
            v = lv.set_index("model").loc[k]
            x["level"], x["status"] = v["verdict"], "%s; уровень по V14: %s; уровень 4 не проверялся" % (STATUS, v["verdict"])
        rows.append(x)
    rows.append({"model": "HAR-J Panel Hybrid", "config": "-", "status": "не в этом пакете (панель вне 80, Л10; отдельный этап)"})
    return pd.DataFrame(rows)
