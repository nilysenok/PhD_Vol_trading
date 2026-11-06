"""Строки и учёт волны D (поток XGB): метрики из ряда прогнозов (`rowsx.row_of`) + Δ к своей базе, crit_wf, срезы Г4 / stress_RV, реестр trials (wave D-xgb), CHECKPOINT used.temporal_regime <= 30."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s6.xgb import cfgx, critx, d_calc, d_comp, d_space, rowsx

ROWS = d_calc.DATA / "rows_D.json"
TRIALS = cfgx.XGB_REP / "OPTUNA_TRIALS.csv"
CKPT = cfgx.XGB_REP / "CHECKPOINT.json"
STATUS = rowsx.STATUS.replace("волна B", "волна D")
NOTE = "волна D Ж-206, поток XGB (окна, частота, режимы); " + STATUS
CEILING = "потолок: кандидаты базы x2_best_j видели 2017–2021 (колонки 2017–2021 не честные)"


def write_csv(df: pd.DataFrame, path, note: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (NOTE, note))
        df.to_csv(f, index=False)


def load_rows() -> dict:
    return json.loads(ROWS.read_text()) if ROWS.exists() else {}


def save_row(r: dict) -> None:
    allr = load_rows()
    allr[r["id"]] = r
    d_calc.DATA.mkdir(parents=True, exist_ok=True)
    ROWS.write_text(json.dumps(allr, ensure_ascii=False, indent=1, default=float))


def g4_series() -> pd.Series:
    return pd.read_parquet(common.OUT / "stress_p3.parquet").set_index("d")["stress"]


def slices(fc: pd.DataFrame, rowidx: np.ndarray, st_rv: np.ndarray, trans: np.ndarray | None = None) -> list:
    """Срезы QLIKE: стресс/спокойно по Г4 (`stress_p3`, состояние на d) и по stress_RV (строка d); trans — маска переходов R4 по датам сетки."""
    q = np.asarray(metrics.qlike(fc["rv"].to_numpy(float), fc["sigma2"].to_numpy(float)), float)
    g4 = g4_series().reindex(fc["d"]).to_numpy(bool)
    rv_st = np.asarray(st_rv)[rowidx]
    sl = [("стресс Г4", g4), ("спокойно Г4", ~g4), ("стресс_RV", rv_st), ("спокойно_RV", ~rv_st)]
    if trans is not None:
        sl += [("переход (смена состояния + 5 дн.)", trans), ("вне перехода", ~trans)]
    return [{"slice": n, "n": int(m.sum()), "qlike": float(q[m].mean()) if m.any() else float("nan")} for n, m in sl]


def row_d(name: str, tid: str, ev: dict, ctx: dict, base: dict, kind: str, st_rv: np.ndarray, direction: str, extra: dict | None = None,
          trans: np.ndarray | None = None) -> dict:
    """ev: fc, folds, raw. base: s2 (σ̂² своей базы), crit_pen, crit_wf, name. ctx: опора ctrl_f4 для rowsx.row_of."""
    c = cfgx.mk(name, "l2", direction=direction)
    r = rowsx.row_of(c, tid, ev["fc"], ev["folds"], ctx)
    r.update({k.replace("_f4", "_base"): v for k, v in rowsx.with_f4(ev["fc"], ctx["ctrl"], base["s2"]).items()})
    q = metrics.qlike(ev["fc"]["rv"].to_numpy(float), ev["fc"]["sigma2"].to_numpy(float))
    r["crit_wf"] = d_comp.crit_wf(q, ev["fc"]["target_date"])
    r.update(kind=kind, base=base["name"], d_crit_pen=r["crit_pen"] - base["crit_pen"], d_crit_wf=r["crit_wf"] - base["crit_wf"],
             beats_base=bool(critx.better(r["crit_pen"], base["crit_pen"]) and not r["unstable"]), train_secs=float(ev["raw"]["secs"].sum()),
             slices=slices(ev["fc"], ev["raw"]["row"].to_numpy(), st_rv, trans), status=STATUS + ("; " + CEILING if base["name"] == "x2_best_j" else ""))
    r["source"] = "волна D Ж-206 (новое)"
    r.update(extra or {})
    return r


def base_info(name: str, ev: dict) -> dict:
    q = metrics.qlike(ev["fc"]["rv"].to_numpy(float), ev["fc"]["sigma2"].to_numpy(float))
    return {"name": name, "s2": ev["fc"]["sigma2"].to_numpy(float), "crit_pen": critx.crit(ev["folds"])["crit_pen"], "crit_wf": d_comp.crit_wf(q, ev["fc"]["target_date"])}


def trial_rec(tid: str, canon: dict, direction: str, inner: str, secs: float, stamp: str) -> dict:
    return {"trial_id": tid, "wave": "D-xgb", "direction": direction, "config": str(canon), "metric_inner": inner, "seed": 42, "time": "%s;%.1fs" % (stamp, secs)}


def inner_of(r: dict) -> str:
    keys = ("qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025", "d_harj_no_240222", "d_base", "d_f4")
    return "crit_pen=%.6f;crit_wf=%.6f;d_crit_pen=%.6f;%s" % (r["crit_pen"], r["crit_wf"], r["d_crit_pen"], ";".join("%s=%.6f" % (k, r[k]) for k in keys))


def n_trials() -> int:
    if not TRIALS.exists() or not TRIALS.stat().st_size:
        return 0
    return int((pd.read_csv(TRIALS)["wave"] == "D-xgb").sum())


def register(recs: list) -> int:
    """Дописывает только новые trial_id; превышение лимита 30 — ошибка до записи; повтор запуска не удваивает."""
    old = set(pd.read_csv(TRIALS)["trial_id"]) if TRIALS.exists() and TRIALS.stat().st_size else set()
    new = {r["trial_id"] for r in recs} - old
    if n_trials() + len(new) > d_space.LIMIT:
        raise ValueError("бюджет temporal_regime: %d + %d > %d" % (n_trials(), len(new), d_space.LIMIT))
    n = cfgx.register(TRIALS, recs)
    sync_checkpoint()
    return n


def sync_checkpoint() -> dict:
    n = n_trials()
    if n > d_space.LIMIT:
        raise ValueError("бюджет temporal_regime: %d > %d" % (n, d_space.LIMIT))
    ck = json.loads(CKPT.read_text())
    ck["used"]["temporal_regime"] = n
    CKPT.write_text(json.dumps(ck, ensure_ascii=False, indent=1))
    return ck


def short(r: dict) -> str:
    return "%s crit_pen %.5f (Δ %+.5f) crit_wf %.5f qlike %.4f 2017-21 %.4f 2022 %.4f persp %s unstable %s" % (
        r["id"], r["crit_pen"], r["d_crit_pen"], r["crit_wf"], r["qlike_all"], r["qlike_2017_2021"], r["qlike_2022"], r["perspective"], r["unstable"])
