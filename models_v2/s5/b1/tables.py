"""Сборка таблиц B1 (OBJECTIVE_EXPERIMENTS / HAR_BOOSTING_RESULTS / REGIME_MODEL_RESULTS) и реестра OPTUNA_TRIALS_B1.csv (только дописывание)."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common, registry
from models_v2.s5.b1 import configs, evalrows, rowsfc, world

REG_B1 = common.REP / "OPTUNA_TRIALS_B1.csv"
CONTROLS = ("ctrl_b2", "ctrl_harj", "ctrl_b1")


def all_rows(w, specs, procs: int = 4) -> tuple:
    """-> (ctrl, {row_id: (fc, extra, ci|None)}, folds, info)."""
    ctrl = common.controls()
    rows, folds, _ = rowsfc.stream_rows(w, specs, ctrl, procs)
    p = rowsfc.pilot_b1(ctrl)
    info = {"rel_ctrl_b2": rowsfc.check_ctrl(w, rows["ctrl_b2"][0], ctrl, "s2_b2"),
            "rel_ctrl_b1": rowsfc.check_ctrl(w, rows["ctrl_b1"][0], pd.DataFrame({"d": p["d"], "rv": p["rv"], "s2_b1": p["sigma2"]}), "s2_b1")}
    rows["ctrl_harj"] = (rowsfc.streams.frame_of(w, w.grid, ctrl["s2_harj"].to_numpy(float)), {}, None)
    for k, v in rowsfc.h3_rows(w, specs, ctrl).items():
        rows[k] = v + (None,)
    for k, v in rowsfc.r2_rows(w, specs, ctrl, procs).items():
        rows[k] = v + (None,)
    rows["reg_r3"] = rowsfc.r3_row(w, specs, ctrl, procs) + (None,)
    best = rowsfc.best_h2(folds, rows)
    info["best_h2"] = best
    for k, v in rowsfc.r4_rows(w, ctrl, rows[best][0]["sigma2"].to_numpy(float)).items():
        rows[k] = v + (None,)
    for rid, (fc, _, _) in rows.items():
        fc.to_parquet(world.B1_DIR / ("fc_%s.parquet" % rid))
    return ctrl, rows, folds, info


def q_int(rid: str, rows: dict, folds: dict) -> tuple:
    """(вид, значение) внутреннего критерия строки: LOYO — одномодельные потоки, иначе walk-forward."""
    cfg = configs.BY_ID[rid]
    if cfg["kind"] == "stream":
        f = folds[cfg["stream"]]
        return "loyo", evalrows.q_loyo(f[f.ci == rows[rid][2]])
    return "wf", q_wf_of(rows[rid][0])


def q_wf_of(fc: pd.DataFrame) -> float:
    return evalrows.q_wf(metrics.qlike(fc["rv"], fc["sigma2"]), fc["target_date"])


def table(ctrl, rows: dict, folds: dict) -> pd.DataFrame:
    kinds = {k: q_int(k, rows, folds) for k in rows}
    b2 = {"loyo": kinds["ctrl_b2"][1], "wf": q_wf_of(rows["ctrl_b2"][0])}
    out = []
    for cfg in configs.ROWS:
        rid = cfg["row_id"]
        fc, extra, _ = rows[rid]
        r = evalrows.row(fc, ctrl, extra)
        kind, q = kinds[rid]
        r.update({"trial_id": configs.trial_id(cfg) if cfg["trial"] else "", "row_id": rid, "direction": cfg["direction"] or "control",
                  "desc": cfg["desc"], "q_int_kind": kind, "q_int": q, "q_int_b2": b2[kind], "q_loyo": q if kind == "loyo" else np.nan})
        r["perspective"] = evalrows.perspective(q, b2[kind], r["d_harj_no_240222"]) if cfg["trial"] else ""
        out.append(r)
    lead = ["trial_id", "row_id", "direction", "desc", "q_int_kind", "q_int", "q_int_b2", "q_loyo", "q_wf", "perspective"]
    df = pd.DataFrame(out)
    return df[lead + [c for c in df.columns if c not in lead]]


def write_tables(df: pd.DataFrame) -> None:
    head = "# разработка; волна B1 Ж-204; до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER, не независимый OOS); код+входы %s\n" % world.key()
    for dirn, fn in configs.TABLE.items():
        part = df[df.direction.isin([dirn, "control"])]
        with open(common.REP / fn, "w", encoding="utf-8") as f:
            f.write(head)
            part.to_csv(f, index=False)


def append_registry(df: pd.DataFrame) -> int:
    common.REP.mkdir(parents=True, exist_ok=True)
    have = set()
    if REG_B1.exists():
        have = set(pd.read_csv(REG_B1)["trial_id"].astype(str))
    new = not REG_B1.exists()
    n = 0
    with open(REG_B1, "a", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(registry.TRIAL_COLS)
        for cfg in configs.trials():
            tid = configs.trial_id(cfg)
            if tid in have:
                continue
            r = df[df.row_id == cfg["row_id"]].iloc[0]
            desc = {k: float(r[k]) for k in ("qlike_all", "qlike_no_240222", "d_harj_no_240222", "d_b2")}
            wr.writerow([tid, "B1", cfg["direction"], json.dumps(cfg, ensure_ascii=False, sort_keys=True), "%s=%.6f" % (r["q_int_kind"], r["q_int"]),
                         json.dumps(desc), time.strftime("%Y-%m-%d %H:%M:%S"), configs.SEED, ""])
            n += 1
    return n
