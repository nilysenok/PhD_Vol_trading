"""Таблицы потока LGBM: строка описания (колонки Ж-204 волны B + статус), спокойный/стресс, реестр trials (только дописывание), CHECKPOINT."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5.b2 import tab as tab2
from models_v2.s6 import diag, paths6

REP = paths6.REP / "lightgbm"
DATA = paths6.OUT / "lightgbm" / "B"
TRIALS = REP / "OPTUNA_TRIALS.csv"
CHECK = REP / "CHECKPOINT.json"
TRIAL_COLS = ("trial_id", "wave", "direction", "config", "metric_inner", "seed", "time")
STATUS = paths6.STATUS


def calm_stress(fc: pd.DataFrame, base: pd.DataFrame) -> dict:
    """QLIKE модели и разность к baseline отдельно в спокойные / стрессовые дни (стресс Г4 по данным ≤ d)."""
    st = diag.inputs(pd.DataFrame({"d": fc["d"]}))["stress"].to_numpy(bool)
    q, qb = metrics.qlike(fc["rv"], fc["sigma2"]), metrics.qlike(base["rv"], base["sigma2"])
    out = {}
    for lab, m in (("calm", ~st), ("stress", st)):
        out.update({"qlike_%s" % lab: float(q[m].mean()), "d_base_%s" % lab: float((q - qb)[m].mean()), "n_%s" % lab: int(m.sum())})
    return out


def row(name: str, kind: str, fc, har, base, st: dict, meta: dict) -> dict:
    """meta: trial_id, direction, crit_pen, crit_pen_base, crit_mean, worst_fold, desc. Перспективно: crit_pen < crit_pen baseline И Δ к HAR-J(б) без 24.02.2022 ≤ 0."""
    s = dict({"mse_train": np.nan, "mse_block": np.nan}, **st)
    m2 = {"trial_id": meta["trial_id"], "direction": meta["direction"], "crit_loyo": meta["crit_pen"], "crit_loyo_b2": meta["crit_pen_base"],
          "crit_wf_b2": 0.0}
    r = tab2.row(name, fc, har, base, s, m2)
    r = {k.replace("_b2_", "_base_").replace("dloss_b2_", "dloss_base_"): v for k, v in r.items() if k not in ("mse_ratio", "crit_kind", "cheap_ok")}
    r.update(calm_stress(fc, base))
    r.update(kind=kind, crit_pen=meta["crit_pen"], crit_mean=meta.get("crit_mean", np.nan), worst_fold=meta.get("worst_fold", np.nan),
             k_min=st.get("k_min", np.nan), share_k1=st.get("share_k1", np.nan), share_k_cap=st.get("share_k_cap", np.nan),
             n_nonfinite_es=st.get("n_nonfinite_es", np.nan), n_eps_floor=st.get("n_eps_floor", np.nan), status=STATUS, desc=meta.get("desc", ""))
    r.pop("crit_loyo", None)
    r.pop("crit_wf", None)
    return r


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; волна B Ж-206 поток LGBM; %s; критерий crit_pen = среднее годовых LOYO + 0,25·(худший − среднее)\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")


def trial_row(tid: str, direction: str, config: str, metric: str, secs: float) -> dict:
    return {"trial_id": tid, "wave": "B", "direction": direction, "config": config, "metric_inner": metric, "seed": 42, "time": "%.1f" % secs}


def register(rows: list, path: Path = TRIALS) -> int:
    """Дописывает только новые trial_id; -> число добавленных."""
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=TRIAL_COLS)
    new = [r for r in rows if r["trial_id"] not in set(old["trial_id"].astype(str))]
    if new:
        pd.concat([old, pd.DataFrame(new)], ignore_index=True)[list(TRIAL_COLS)].to_csv(path, index=False)
    return len(new)


def checkpoint(path: Path = CHECK, trials: Path = TRIALS) -> dict:
    """Пересчёт счётчиков из реестра: direction 'hp' -> hp, остальные -> obj_feat. Лимиты потока: obj_feat 45, hp 15 из 55 (LB0)."""
    c = json.loads(path.read_text())
    t = pd.read_csv(trials)
    c["used"]["hp"] = int((t["direction"] == "hp").sum())
    c["used"]["obj_feat"] = int((t["direction"] != "hp").sum())
    c["limits_stream_b"] = {"obj_feat": 45, "hp_lb0": 15}
    c["updated"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(c, ensure_ascii=False, indent=1))
    if c["used"]["obj_feat"] > 45 or c["used"]["hp"] > 15:
        raise ValueError("превышен бюджет потока LGBM: %s" % c["used"])
    return c
