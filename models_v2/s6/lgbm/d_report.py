"""Волна D, поток LGBM: итоговые таблицы TEMPORAL_VALIDATION.csv, REGIME_PERFORMANCE_D.csv, D_WINDOWS_CALM_STRESS.csv (из сохранённых рядов; ничего не считается заново)."""
from __future__ import annotations

import pandas as pd

from models_v2.s6.lgbm import d_comp, d_reg, d_run, d_world

REP = d_world.tab.REP


def all_names() -> list:
    return (["D_w0"] + list(d_run.WINS) + ["D_pick_window"] + ["%s_%s" % (d_run.best_window(), s) for s in d_run.FREQ]
            + list(d_reg.R1) + ["D_R4a_switch", "D_R4b_switch_pick"])


def regime_table(W: dict, names: list) -> pd.DataFrame:
    st_g = d_reg.stress_rows(W["F"].dates)[W["grid"]]
    base = d_world.rows(W, names)
    rec = []
    for n, r in zip(names, base.to_dict("records")):
        fc, st = d_world.get(n)
        t = d_comp.transitions(fc, st_g)
        rec.append({"model": n, "direction": st["direction"], "qlike_calm": r["qlike_calm"], "d_base_calm": r["d_base_calm"], "n_calm": r["n_calm"],
                    "qlike_stress": r["qlike_stress"], "d_base_stress": r["d_base_stress"], "n_stress": r["n_stress"], **t,
                    **{k: st.get("switch", {}).get(k, float("nan")) for k in ("n_regime_flips", "n_config_changes", "days_configs_differ", "share_chosen_worse_ex_post")},
                    "qlike_y2022": r["qlike_y2022"], "qlike_no_2022": r["qlike_no_2022"], "status": r["status"]})
    return pd.DataFrame(rec)


def windows_calm_stress(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df["model"].isin(["D_w0"] + list(d_run.WINS) + ["D_pick_window"])].copy()
    e = w.set_index("model").loc["D_w0"]
    out = w[["model", "qlike_calm", "qlike_stress", "qlike_all", "qlike_y2022", "qlike_no_2022", "qlike_no_20220224", "qlike_p2017_2021", "qlike_p2023_2025", "crit_pen"]].copy()
    for c in ("qlike_calm", "qlike_stress", "qlike_all", "qlike_y2022", "qlike_no_2022"):
        out["d_vs_expanding_" + c[6:]] = w[c] - e[c]
    out["rank_calm"], out["rank_stress"] = out["qlike_calm"].rank(method="min"), out["qlike_stress"].rank(method="min")
    return out


def stage_report(W: dict) -> None:
    names = all_names()
    df = d_world.rows(W, ["C0_inh_F4"] + names)
    d_world.write(df, REP / "TEMPORAL_VALIDATION.csv", "все конфигурации волны D (база C0_inh_F4 первой строкой); DM к HAR-J(б), к базе, к lb0_inh (dloss_ref_*)")
    d_world.write(windows_calm_stress(df[df["model"] != "C0_inh_F4"]), REP / "D_WINDOWS_CALM_STRESS.csv", "окна в спокойном и стрессовом режиме (стресс Г4 по данным <= d)")
    d_world.write(regime_table(W, names), REP / "REGIME_PERFORMANCE_D.csv", "режимы: спокойно/стресс, переходы (первые 5 дней), задержка реакции, переключения")
    print(df[["model", "direction", "crit_pen", "crit_wf", "qlike_all", "qlike_y2022", "qlike_no_2022", "qlike_calm", "qlike_stress", "dloss_harj_no_20220224"]].to_string(), flush=True)
    print(d_world.checkpoint(), flush=True)
