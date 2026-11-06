"""Ж-211: полный реестр гипотез (ALL_TRIALS.csv), счётчики бюджета (CHECKPOINT.json), N для поправок.
Прежние реестры только читаются: Ж-204 (sber_h1_deep_v1/OPTUNA_TRIALS.csv), Ж-206/210 (xgboost/, lightgbm/, xgb_lgbm_ensemble/), пилот Ж-200/202 (INVENTORY, семьи p1/p2/p3/pb/fix)."""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from models_v2.s6 import paths6

REP = paths6.ROOT / "docs/otchety/hybrid_final_v1"
OUT = paths6.ROOT / "data/regen/hybrid_final_v1"
ALL = REP / "ALL_TRIALS.csv"
CHECK = REP / "CHECKPOINT.json"
LIMITS = {"residual_xgb": 15, "residual_lgbm": 15, "bounded": 10, "dyn_weights": 10, "regime": 10}
TOTAL = 60
PILOT = ("forecasts_p1", "forecasts_p2", "forecasts_p3", "forecasts_pb", "fix_forecasts_a", "fix_forecasts_c")
COLS = ["key", "program", "source", "direction", "config", "kind"]


def _read(p):
    return pd.read_csv(p, comment="#")


def prior_rows() -> pd.DataFrame:
    rows = []
    d = _read(paths6.OLD_REP / "OPTUNA_TRIALS.csv")
    for _, r in d.iterrows():
        rows.append(("Ж-204:" + str(r["trial_id"]), "Ж-204", "sber_h1_deep_v1/OPTUNA_TRIALS.csv", str(r["направление"]), str(r["конфигурация"])[:160], "trial"))
    for sub in ("xgboost", "lightgbm", "xgb_lgbm_ensemble"):
        d = _read(paths6.REP / sub / "OPTUNA_TRIALS.csv")
        for _, r in d.iterrows():
            rows.append(("Ж-206:%s:%s" % (sub, r["trial_id"]), "Ж-206/210", sub + "/OPTUNA_TRIALS.csv", str(r["direction"]), str(r["config"])[:160], "trial"))
    inv = _read(paths6.REP / "INVENTORY.csv")
    inv = inv[inv["family"].str.split(";").str[0].isin(PILOT)].drop_duplicates("id")
    for _, r in inv.iterrows():
        rows.append(("pilot:" + str(r["id"]), "Ж-200/202", "INVENTORY.csv", str(r["family"]), str(r["id"]), "pilot_forecast"))
    return pd.DataFrame(rows, columns=COLS)


def counts(df: pd.DataFrame) -> dict:
    n204 = int(((df["program"] == "Ж-204")).sum())
    n206 = int((df["program"] == "Ж-206/210").sum())
    npil = int((df["program"] == "Ж-200/202").sum())
    return {"n_zh204": n204, "n_zh206_210": n206, "n_pilot": npil, "N_prior": n204 + n206 + npil, "N_ref_288": n204 + n206}


def tid(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def load() -> pd.DataFrame:
    return pd.read_csv(ALL) if ALL.exists() else pd.DataFrame(columns=COLS)


def init() -> dict:
    """Свести реестры в ALL_TRIALS.csv (новые строки Ж-211 сохраняются)."""
    REP.mkdir(parents=True, exist_ok=True)
    cur = load()
    new = cur[cur["program"] == "Ж-211"] if len(cur) else cur
    df = pd.concat([prior_rows(), new], ignore_index=True)
    if df["key"].duplicated().any():
        raise ValueError("дубль key в реестре: %s" % df.loc[df["key"].duplicated(), "key"].tolist()[:5])
    df.to_csv(ALL, index=False, lineterminator="\n")
    return write_check(df)


def used(df: pd.DataFrame) -> dict:
    n = df[df["program"] == "Ж-211"]
    return {k: int((n["direction"] == k).sum()) for k in LIMITS}


def write_check(df: pd.DataFrame) -> dict:
    c, u = counts(df), used(df)
    c.update({"limits": LIMITS, "limit_total": TOTAL, "used": u, "used_total": sum(u.values()), "N_final": c["N_prior"] + sum(u.values()),
              "updated": time.strftime("%Y-%m-%d %H:%M")})
    CHECK.write_text(json.dumps(c, ensure_ascii=False, indent=1))
    return c


def register(direction: str, cfg: dict) -> str:
    """Новая уникальная конфигурация Ж-211; повтор того же cfg — не новый trial; превышение лимита направления или 60 — ValueError."""
    if direction not in LIMITS:
        raise ValueError("чужое направление %s" % direction)
    df = load()
    key = "Ж-211:%s:%s" % (direction, tid(cfg))
    if key in set(df["key"]):
        return key
    u = used(df)
    if u[direction] + 1 > LIMITS[direction] or sum(u.values()) + 1 > TOTAL:
        raise ValueError("бюджет Ж-211 превышен: %s %s" % (direction, u))
    row = pd.DataFrame([(key, "Ж-211", "hybrid_final_v1", direction, json.dumps(cfg, ensure_ascii=False, sort_keys=True), "trial")], columns=COLS)
    df = pd.concat([df, row], ignore_index=True)
    df.to_csv(ALL, index=False, lineterminator="\n")
    write_check(df)
    return key


def n_final() -> int:
    """N для поправок: полный реестр на момент вызова (Ж-204 + Ж-206/210 + пилот + новые Ж-211)."""
    df = load()
    return counts(df)["N_prior"] + len(df[df["program"] == "Ж-211"])
