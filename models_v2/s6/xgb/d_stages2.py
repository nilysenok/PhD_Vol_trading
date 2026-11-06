"""Стадии волны D потока XGB: d2 (частота), r1 (режимные признаки), r4 (переключение конфигураций), tables."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.xgb import cfgx, d_calc, d_comp, d_regime, d_rows, d_space, worldx
from models_v2.s6.xgb.d_stages import BASE_NAME, run_cfg, setup, stamp

R4_WINDOWS = ("roll1008", "roll504")
TV_COLS = ["id", "kind", "base", "trial_id", "window_pairs", "half_life_rows", "freq", "r1", "crit_pen", "d_crit_pen", "crit_wf", "d_crit_wf", "qlike_all", "qlike_2017_2021",
           "qlike_2022", "qlike_2023_2025", "qlike_2026", "qlike_no_240222", "qlike_no_2022", "mean_r", "d_harj", "p_harj", "d_harj_no_240222", "p_harj_no_240222",
           "d_f4", "p_f4", "d_f4_no_240222", "p_f4_no_240222", "d_base", "p_base", "d_base_no_240222", "p_base_no_240222", "share_years_win_harj", "beats_base",
           "perspective", "unstable", "train_secs", "n_dates", "n_switches", "n_shocks", "share_flagged", "median_delay", "share_same_day", "choice", "status"]


def best_window(rows: dict, b: str, base_crit: float) -> str:
    """argmin crit_pen среди expanding и шести строк D1 базы (фиксированное окно, не annual); ничья — expanding."""
    best, bw = base_crit, "expanding"
    for r in rows.values():
        if r.get("kind") == "d1_window" and r["base"] == BASE_NAME[b] and r["crit_pen"] < best - 1e-12:
            best, bw = r["crit_pen"], r["id"].split("_")[1]
    return bw


def stage_d2(w: dict) -> int:
    """D2: на лучшем окне каждой базы еженедельно и ежемесячно (≤ 4 trials); правило Q4 по crit_wf."""
    st, dec = setup(w), {}
    for b in d_space.BASES:
        rows = d_rows.load_rows()
        win = best_window(rows, b, st["bases"][b]["crit_pen"])
        n, hl = d_space.WINDOWS[win]
        daily = st["bases"][b] if win == "expanding" else rows[d_space.name_of(d_space.DCfg(b, win))]
        cwf, secs = {"d": daily["crit_wf"]}, {"d": st["ex"][b]["raw"]["secs"].sum() if win == "expanding" else daily["train_secs"]}
        for fq in ("w", "m"):
            _, r = run_cfg(w, st, d_space.DCfg(b, win, fq), "d2_freq", "temporal", {"window_pairs": n, "half_life_rows": hl, "freq": fq})
            cwf[fq], secs[fq] = r["crit_wf"], r["train_secs"]
        dec[BASE_NAME[b]] = {"window": win, "crit_wf": cwf, "train_secs": secs, "chosen": d_comp.freq_choice(cwf), "threshold": d_space.FREQ_THRESHOLD}
    paths6.dump_json(dec, cfgx.XGB_REP / "FREQ_DECISION_D.json")
    print("D2:", json.dumps(dec, default=float)[:700])
    return 0


def stage_r1(w: dict) -> int:
    """D3-R1: режимные признаки (q_w | q_w, stress_RV, переход) в модель, expanding, ежедневно, обе базы (4 trials)."""
    st = setup(w)
    for b in d_space.BASES:
        for r1 in ("qw", "all"):
            run_cfg(w, st, d_space.DCfg(b, "expanding", "d", r1), "r1", "regime", {"r1": r1, "freq": "d"})
    print("R1: trials D = %d" % d_rows.n_trials())
    return 0


def stage_r4(w: dict) -> int:
    """D3-R4: стресс_RV → rolling W, спокойно → expanding (W = 1008, 504); без новых обучений; 4 trials (составные ряды)."""
    st = setup(w)
    F, grid, state = w["F"], w["grid"], st["st"]
    trans, sw = d_regime.transition_mask(state[grid]), d_regime.n_switches(state[grid])
    delay = d_regime.reaction_delay(state, d_regime.shock_rows(F.rv))
    for b in d_space.BASES:
        for W in R4_WINDOWS:
            a, c = d_calc.evaluate(w, d_space.DCfg(b, W)), st["ex"][b]
            raw = d_comp.compose_raw(a["raw"], c["raw"], state)
            ev = {"raw": raw, "folds": d_comp.compose_folds(a["long"], c["long"], state), "fc": worldx.sigma_frame(w, d_space.spec_for(w), raw)}
            canon = {"alg": "xgboost", "wave": "D", "kind": "r4", "base": b, "stress_cfg": a["tid"], "calm_cfg": c["tid"], "state": "stress_RV (RV_w > q67 по 252 предыдущим)"}
            tid = d_space.digest(canon)
            r = d_rows.row_d("%s_r4_%s" % (b, W), tid, ev, st["ctx"], st["bases"][b], "r4", state, "regime", {"n_switches": sw, **delay, "stress_cfg": a["name"], "calm_cfg": c["name"]}, trans)
            d_rows.save_row(r)
            d_rows.register([d_rows.trial_rec(tid, canon, "regime", d_rows.inner_of(r), ev["raw"]["secs"].sum(), stamp())])
            print(d_rows.short(r), flush=True)
    print("R4: trials D = %d; переключений %d; %s" % (d_rows.n_trials(), sw, delay))
    return 0


def stage_tables() -> int:
    rows = d_rows.load_rows()
    tv = pd.DataFrame(list(rows.values())).reindex(columns=TV_COLS)
    reg = pd.DataFrame([{"id": r["id"], "kind": r["kind"], "base": r["base"], **s} for r in rows.values() for s in r["slices"]])
    d_rows.write_csv(tv, cfgx.XGB_REP / "TEMPORAL_VALIDATION.csv", "все строки D: опоры, окна/веса (D1), годовой выбор окна (процедура), частота (D2), R1, R4; Δ к своей базе и к HAR-J(б); " + d_rows.CEILING)
    d_rows.write_csv(reg, cfgx.XGB_REP / "REGIME_PERFORMANCE_D.csv", "QLIKE по срезам: стресс/спокойно по Г4 (stress_p3) и по stress_RV, переходы R4")
    ck = d_rows.sync_checkpoint()
    reg_ids = pd.read_csv(d_rows.TRIALS)
    dup = tv["id"].duplicated().any() or reg_ids["trial_id"].duplicated().any()
    print("tables: TEMPORAL_VALIDATION %d строк, REGIME_PERFORMANCE_D %d строк; used.temporal_regime = %d; дубли: %s" % (len(tv), len(reg), ck["used"]["temporal_regime"], "ЕСТЬ" if dup else "нет"))
    return 1 if dup else 0
