"""Волна B, поток XGB: запуск `python3 models_v2/s6/xgb/runx.py --stage <control|stab|trials|cond|tables|check>`. Процессов <= 4. Прежние каталоги — только чтение."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2.s4 import rows as s4rows  # noqa: E402
from models_v2.s5 import common  # noqa: E402
from models_v2.s5.b1 import jobs  # noqa: E402
from models_v2.s6 import paths6  # noqa: E402
from models_v2.s6.xgb import calcx, cfgx, critx, rowsx, stabx, worldx  # noqa: E402

ROWS_JSON = calcx.DATA / "rows_B.json"
TRIALS = cfgx.XGB_REP / "OPTUNA_TRIALS.csv"
CKPT = cfgx.XGB_REP / "CHECKPOINT.json"


def load_rows() -> dict:
    return json.loads(ROWS_JSON.read_text()) if ROWS_JSON.exists() else {}


def save_row(r: dict) -> None:
    allr = load_rows()
    allr[r["id"]] = r
    ROWS_JSON.write_text(json.dumps(allr, ensure_ascii=False, indent=1, default=float))


def control_dates_ok(w: dict) -> dict:
    """P4: ctrl_f4 на 3 датах побитно = fc_b2_f4.parquet (прежний каталог, чтение)."""
    ref = pd.read_parquet(common.DEEP / "b2" / "fc_b2_f4.parquet").set_index("d")["sigma2"]
    spec, F, out = worldx.spec_of(w, cfgx.CTRL), w["F"], {}
    for ds in paths6.CONTROL_DATES:
        d = int(np.flatnonzero(F.dates == np.datetime64(ds))[0])
        m = jobs.fit_at(F, spec, jobs.pairs(F, spec, d), w["i2"][s4rows.year_of(F, d)])
        s2 = float(np.exp(spec.off[d] + m["predict"](spec.X[[d]])[0] + m["s2_es"] / 2))
        out[ds] = {"new": s2, "old": float(ref.loc[pd.Timestamp(ds)]), "bitwise": s2 == float(ref.loc[pd.Timestamp(ds)])}
    return out


def ctx_of(w: dict) -> dict:
    """Опора: crit Б-2 (фолды Ж-204, чтение) и ctrl_f4 (кэш этапа control); σ̂² опоры."""
    fb = pd.read_parquet(worldx.FOLDS_B2)
    fc, fl, _, _ = calcx.evaluate(w, cfgx.CTRL)
    return {"ctrl": w["ctrl"], "f4_s2": fc["sigma2"].to_numpy(float), "crit_b2": critx.crit(fb[fb["ci"] == 0]), "crit_f4": critx.crit(fl)}


def stage_control(w: dict) -> int:
    dates = control_dates_ok(w)
    fc, fl, _, secs = calcx.evaluate(w, cfgx.CTRL)
    ref = pd.read_parquet(common.DEEP / "b2" / "fc_b2_f4.parquet")
    rel = float(np.max(np.abs(fc["sigma2"].to_numpy() / ref["sigma2"].to_numpy() - 1)))
    ctx = ctx_of(w)
    st = json.loads((common.DEEP / "b2" / "st_b2_f4.json").read_text())
    res = {"dates": dates, "full_grid_max_rel_diff": rel, "crit_mean_f4": ctx["crit_f4"]["crit_mean"], "crit_mean_f4_old": st["crit_loyo"],
           "crit_f4": ctx["crit_f4"], "crit_b2": ctx["crit_b2"], "secs": secs}
    ok = all(v["bitwise"] for v in dates.values()) and abs(res["crit_mean_f4"] - st["crit_loyo"]) < 1e-9 and rel < 1e-9
    r = rowsx.row_of(cfgx.CTRL, cfgx.prior_trial_ids()[cfgx.key(cfgx.CTRL)][1] + "#ctrl", fc, fl, ctx)
    r["source"], r["objective"] = "контроль: Б-2+F4 в конвейере волны B (не trial; trial_id Ж-204)", "L2, ln-цель"
    save_row(r)
    paths6.dump_json(res, calcx.DATA / "CONTROL_B.json")
    paths6.dump_json(res, cfgx.XGB_REP / "CONTROL_B.json")
    print("B2:", "OK" if ok else "ПРОВАЛ", json.dumps(res, default=float)[:600])
    return 0 if ok else 1


def run_trial(w: dict, c: dict, ctx: dict) -> dict | None:
    """Новая конфигурация -> счёт, строка, реестр. Дубль Ж-204 не считается trial (None)."""
    dup = cfgx.is_duplicate(c)
    if dup:
        print("дубль Ж-204: %s -> %s %s (не trial)" % (c["name"], dup[0], dup[1]))
        return None
    tid = cfgx.trial_id(c)
    fc, fl, _, secs = calcx.evaluate(w, c)
    r = rowsx.row_of(c, tid, fc, fl, ctx)
    save_row(r)
    cfgx.register(TRIALS, [rowsx.trial_record(c, tid, r, secs, time.strftime("%Y-%m-%d %H:%M"))])
    cfgx.sync_checkpoint(CKPT, TRIALS)
    print(c["name"], tid, "qlike %.4f crit_pen %.5f persp %s beats_f4 %s unstable %s (%.0fс)" % (r["qlike_all"], r["crit_pen"], r["perspective"],
                                                                                              r["beats_f4_ref"], r["unstable"], secs), flush=True)
    return r


def stage_trials(w: dict) -> int:
    ctx = ctx_of(w)
    for c in cfgx.NEW:
        run_trial(w, c, ctx)
    return 0


def decide_cond(rows: dict) -> dict:
    """Правило GATES_B_xgb: объектив-кандидат — лучший crit_pen среди x3/x4/x5 с beats_f4_ref; f4_f2 должна иметь beats_f4_ref."""
    objs = {k: v for k, v in rows.items() if k.startswith(("x3_", "x4_", "x5_")) and v["beats_f4_ref"]}
    f2 = rows.get("f4_f2", {}).get("beats_f4_ref", False)
    best = min(objs, key=lambda k: objs[k]["crit_pen"]) if objs else None
    return {"best_obj": best, "f4_f2_beats": bool(f2), "run_cond_obj": bool(best and f2), "run_f1_f2": bool(f2)}


def stage_cond(w: dict) -> int:
    rows = load_rows()
    dec = decide_cond(rows)
    paths6.dump_json(dec, cfgx.XGB_REP / "COND_DECISION.json")
    ctx = ctx_of(w)
    if dec["run_cond_obj"]:
        b = rows[dec["best_obj"]]
        c = cfgx.mk("cond_%s_f4_f2" % dec["best_obj"], b["objective"], b["weights_gamma"], ("F4", "F2"), direction="feature", cond=True)
        run_trial(w, c, ctx)
    if dec["run_f1_f2"]:
        run_trial(w, cfgx.COND[0], ctx)
    print("COND:", dec)
    return 0


def stage_stab(w: dict) -> int:
    y, F = stabx.fd_grid()
    fd = stabx.fd_error(y, F)
    df = stabx.compare_dates(w)
    pr = stabx.pair_diff(df)
    rowsx.write_csv(df, cfgx.XGB_REP / "STABILITY_QLIKE.csv", "кастомный QLIKE против reg:gamma, 12 дат; fd_error=%.3e (допуск %.0e)" % (fd, stabx.TOL_FD))
    rowsx.write_csv(pr.reset_index(), cfgx.XGB_REP / "STABILITY_QLIKE_PAIRS.csv", "|Δ ln σ̂²| и Δk между QLIKE и gamma")
    ok = fd <= stabx.TOL_FD and bool(df["finite"].all())
    print("B3:", "OK" if ok else "ПРОВАЛ", "fd_error %.3e; конечных %d/%d; |Δln| медиана %.4f макс %.4f; Δk медиана %.1f" % (
        fd, int(df["finite"].sum()), len(df), pr["abs_dln"].median(), pr["abs_dln"].max(), pr["dk"].median()))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("control", "stab", "trials", "cond", "tables", "check"))
    a = ap.parse_args()
    if a.stage in ("tables", "check"):
        from models_v2.s6.xgb import checkx, tablex
        return (tablex if a.stage == "tables" else checkx).stage(load_rows())
    paths6.mkdirs()
    w = worldx.load()
    return {"control": stage_control, "stab": stage_stab, "trials": stage_trials, "cond": stage_cond}[a.stage](w)


if __name__ == "__main__":
    sys.exit(main())
