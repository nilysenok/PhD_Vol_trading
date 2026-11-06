"""C-X2 (GATES_C_xgb Q3, Q4): решение по осям с эффектом, совместный Optuna (TPE seed 42, <= 20 trials, цель J — только Y 2017–2021), строки x2_best_j и x2_annual."""
from __future__ import annotations

import json
import time

import optuna

from models_v2.s6 import paths6
from models_v2.s6.xgb import c_calc, c_rows, c_sel, c_space, c_stages, cfgx, critx

N_TRIALS, N_STARTUP, SEED = 20, 8, 42


def axis_deltas(rows: dict) -> dict:
    """{ось: min Δcrit_pen по точкам X1 (дубли опоры пропускаются)}; все 9 осей X1 должны быть в таблице."""
    d = {}
    for r in rows.values():
        if r.get("kind") == "x1" and not r.get("duplicate"):
            d[r["axis"]] = min(d.get(r["axis"], float("inf")), r["d_crit_pen"])
    n = sum(1 for r in rows.values() if r.get("kind") == "x1")
    if n != len(c_space.x1_points()):
        raise SystemExit("стоп: X1 не завершён (%d из %d строк)" % (n, len(c_space.x1_points())))
    return d


def run_study(w: dict, eff: list) -> dict:
    """20 trials по осям эффекта (u непрерывно в диапазоне оси); только фолды; -> {trial_id: {moves, J, crit_pen, folds}}."""
    out = {}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        moves = {ax: round(trial.suggest_float(ax, *c_space.AXES[ax][5]), 4) for ax in eff}
        ev = c_calc.evaluate(w, "l2", moves, daily=False)
        cr, j = critx.crit(ev["folds"]), c_sel.j_guide(ev["folds"])
        out[ev["tid"]] = {"moves": moves, "J": j, "crit_pen": cr["crit_pen"], "folds": ev["folds"]}
        c_rows.register([c_rows.trial_rec(ev["tid"], "l2", moves, "J_guide=%.6f;crit_pen=%.6f;crit_mean=%.6f" % (j, cr["crit_pen"], cr["crit_mean"]),
                                          ev["secs"], time.strftime("%Y-%m-%d %H:%M"), kind="x2")])
        print("x2 trial %d %s J %.5f crit_pen %.5f" % (trial.number, moves, j, cr["crit_pen"]), flush=True)
        return j
    study.optimize(obj, n_trials=N_TRIALS, n_jobs=1)
    return out


def annual_row(w: dict, state: tuple, trials: dict) -> dict:
    """x2_annual: годовой PIT-выбор по pen_Y среди {И2} ∪ кандидатов; процедура (не trial)."""
    ctx, fams, bases = state
    choice, comp = c_sel.annual_select(bases["l2"]["folds"], {t: v["folds"] for t, v in trials.items()})
    base = c_calc.hpc_base(w)
    hpy = {Y: (c_space.apply(base, trials[choice[Y]]["moves"])[Y] if choice[Y] else base[Y]) for Y in choice}
    ev = c_calc.evaluate(w, "l2", {}, folds=False, hpy=hpy, tid="x2a" + c_sel.choice_id("l2", choice))
    ev["folds"] = comp
    extra = {"choice": json.dumps({str(Y): (v or "I2") for Y, v in choice.items()}), "years_not_i2": sum(1 for v in choice.values() if v)}
    r = c_rows.row_c("x2_annual", "l2", {}, ev, ctx, fams["l2"], "x2_annual", extra)
    c_rows.save_row(r)
    print(c_rows.short(r))
    return r


def stage_x2(w: dict) -> int:
    rows = c_rows.load_rows()
    deltas = axis_deltas(rows)
    eff = c_sel.effect_axes(deltas)
    dec = {"deltas_min_crit_pen_by_axis": deltas, "effect_axes": eff, "cap": c_space.CAP_AXES, "status": "DEVELOPMENT"}
    if not eff:
        dec["decision"] = "ABANDON: C-X2 — ни одна ось X1 не дала Δcrit_pen < 0 (все min Δ >= 0): совместный подбор не имеет опоры"
        paths6.dump_json(dec, cfgx.XGB_REP / "X2_DECISION.json")
        print(dec["decision"])
        return 0
    state = c_stages.setup(w)
    trials = run_study(w, eff)
    best = min(trials, key=lambda t: (trials[t]["J"], t))
    dec.update(decision="RUN", best_trial_id=best, best_moves=trials[best]["moves"], best_J=trials[best]["J"], n_unique_trials=len(trials))
    paths6.dump_json(dec, cfgx.XGB_REP / "X2_DECISION.json")
    ctx, fams, _ = state
    ev = c_calc.evaluate(w, "l2", trials[best]["moves"])
    r = c_rows.row_c("x2_best_j", "l2", trials[best]["moves"], ev, ctx, fams["l2"], "x2_best_j", {"J_guide": trials[best]["J"]})
    c_rows.save_row(r)
    print(c_rows.short(r))
    annual_row(w, state, trials)
    return 0
