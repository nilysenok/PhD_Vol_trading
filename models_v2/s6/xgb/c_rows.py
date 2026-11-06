"""Строки и учёт волны C (потока XGB): метрики из ряда прогнозов (rowsx.row_of) + Δ к опоре своей семьи, реестр trials (wave C-xgb), CHECKPOINT used.hp."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models_v2.s6.xgb import c_calc, c_space, cfgx, critx, rowsx, worldx

LIMIT_HP = 55
ROWS = c_calc.DATA / "rows_C.json"
TRIALS = cfgx.XGB_REP / "OPTUNA_TRIALS.csv"
CKPT = cfgx.XGB_REP / "CHECKPOINT.json"
STATUS = rowsx.STATUS.replace("волна B", "волна C")
NOTE = "волна C Ж-206, поток XGB (гиперпараметры); " + STATUS


def write_csv(df: pd.DataFrame, path, note: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (NOTE, note))
        df.to_csv(f, index=False)


def load_rows() -> dict:
    return json.loads(ROWS.read_text()) if ROWS.exists() else {}


def save_row(r: dict) -> None:
    allr = load_rows()
    allr[r["id"]] = r
    c_calc.DATA.mkdir(parents=True, exist_ok=True)
    ROWS.write_text(json.dumps(allr, ensure_ascii=False, indent=1, default=float))


def context(w: dict, base_l2: dict) -> dict:
    """Опора ctrl_f4 для rowsx.row_of: σ̂² и фолды нулевого сдвига l2 (подтверждены побитно к кэшу B на этапе control), crit Б-2 (чтение фолдов Ж-204)."""
    fb = pd.read_parquet(worldx.FOLDS_B2)
    return {"ctrl": w["ctrl"], "f4_s2": base_l2["fc"]["sigma2"].to_numpy(float), "crit_b2": critx.crit(fb[fb["ci"] == 0]), "crit_f4": critx.crit(base_l2["folds"])}


def family_base(ev: dict) -> dict:
    """Опора семьи (l2 — ctrl_f4, gamma — x3_gamma_f4): σ̂² и crit_pen."""
    return {"s2": ev["fc"]["sigma2"].to_numpy(float), "crit_pen": critx.crit(ev["folds"])["crit_pen"]}


def row_c(name: str, family: str, moves: dict, ev: dict, ctx: dict, fam: dict, kind: str, extra: dict | None = None) -> dict:
    """Строка C: rowsx.row_of (метрики, crit, DM к HAR-J/Б-2/ctrl_f4) + Δ к опоре семьи (d_base*, dm_base*, p_base*, d_crit_pen) + оси/сдвиги."""
    c = cfgx.mk(name, "gamma" if family == "gamma" else "l2", direction="hyperparameters")
    r = rowsx.row_of(c, ev["tid"], ev["fc"], ev["folds"], ctx)
    for k, v in rowsx.with_f4(ev["fc"], ctx["ctrl"], fam["s2"]).items():
        r[k.replace("_f4", "_base")] = v
    r.update(family=family, moves=json.dumps(moves, sort_keys=True), kind=kind, d_crit_pen=r["crit_pen"] - fam["crit_pen"],
             beats_base=bool(critx.better(r["crit_pen"], fam["crit_pen"]) and not r["unstable"]), source="волна C Ж-206 (новое)", status=STATUS)
    r.update(extra or {})
    return r


def trial_rec(tid: str, family: str, moves: dict, inner: str, secs: float, stamp: str, kind: str = "hp") -> dict:
    return {"trial_id": tid, "wave": "C-xgb", "direction": "hyperparameters", "config": str(c_space.canon(family, moves, kind)), "metric_inner": inner,
            "seed": 42, "time": "%s;%.1fs" % (stamp, secs)}


def register(rows: list) -> int:
    """Дописывает только новые trial_id; возвращает число добавленных (повтор запуска не удваивает)."""
    n = cfgx.register(TRIALS, rows)
    sync_checkpoint()
    return n


def n_hp_trials() -> int:
    if not TRIALS.exists() or not TRIALS.stat().st_size:
        return 0
    return int((pd.read_csv(TRIALS)["wave"] == "C-xgb").sum())


def sync_checkpoint() -> dict:
    """used.hp = число строк реестра волны C (только новые уникальные); остальные счётчики не трогаются; > 55 — ошибка."""
    n = n_hp_trials()
    if n > LIMIT_HP:
        raise ValueError("бюджет HP: %d > %d" % (n, LIMIT_HP))
    ck = json.loads(CKPT.read_text())
    ck["used"]["hp"] = n
    CKPT.write_text(json.dumps(ck, ensure_ascii=False, indent=1))
    return ck


def short(r: dict) -> str:
    return "%s crit_pen %.5f (Δ %+.5f) qlike %.4f 2017-21 %.4f 2022 %.4f persp %s unstable %s" % (
        r["id"], r["crit_pen"], r["d_crit_pen"], r["qlike_all"], r["qlike_2017_2021"], r["qlike_2022"], r["perspective"], r["unstable"])
