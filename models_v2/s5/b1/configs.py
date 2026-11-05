"""Строки B1 (GATES_B1.md, «Конфигурации первого прохода»): 28 trials + контроли. Trial id = sha256(JSON конфигурации)[:16]."""
from __future__ import annotations

import hashlib
import json

OBJ, HAR, REG = "objectives_target", "har_boosting", "regime_aware"
LIMITS = {OBJ: 17, HAR: 25, REG: 25}
TABLE = {OBJ: "OBJECTIVE_EXPERIMENTS.csv", HAR: "HAR_BOOSTING_RESULTS.csv", REG: "REGIME_MODEL_RESULTS.csv"}
SEED = 42


def _r(row_id, direction, kind, desc, **kw):
    return {"row_id": row_id, "direction": direction, "kind": kind, "desc": desc, "trial": direction is not None, **kw}


ROWS = [
    _r("ctrl_b2", None, "stream", "Б-2 (L2, ln-цель, s²/2) — контроль", stream="ctrl_b2", c=1.0),
    _r("ctrl_harj", None, "harj", "HAR-J(б) — контроль"),
    _r("ctrl_b1", None, "stream", "Б-1 пилота = H2-L2, c = 1 — контроль", stream="ctrl_b1", c=1.0),
    _r("obj_l1n", OBJ, "stream", "L1 MSE по уровню RV/RV_m", stream="obj_l1n", c=1.0),
    _r("obj_l1raw", OBJ, "stream", "L1 MSE по уровню RV (%²) без нормировки", stream="obj_l1raw", c=1.0),
    _r("obj_l3x", OBJ, "stream", "L3 XGBoost reg:gamma", stream="obj_l3x", c=1.0),
    _r("obj_l3l", OBJ, "stream", "L3 LightGBM gamma", stream="obj_l3l", c=1.0),
    _r("obj_l4", OBJ, "stream", "L4 прямой QLIKE (F = ln σ̂², grad/hess)", stream="obj_l4", c=1.0),
    _r("obj_l5a", OBJ, "stream", "L5 L2 с весами RV_w^−0,5", stream="obj_l5a", c=1.0),
    _r("obj_l5b", OBJ, "stream", "L5 L2 с весами RV_w^−1", stream="obj_l5b", c=1.0),
    _r("obj_l5c", OBJ, "stream", "L5 L4 с весами RV_w^−0,5", stream="obj_l5c", c=1.0),
    _r("har_h2_l2_c075", HAR, "stream", "H2-L2, σ̂² = exp(lb + 0,75 f + s²/2)", stream="ctrl_b1", c=0.75),
    _r("har_h2_l2_c05", HAR, "stream", "H2-L2, c = 0,5", stream="ctrl_b1", c=0.5),
    _r("har_h2_l4_c1", HAR, "stream", "H2-L4 (остаток к HAR, QLIKE), c = 1", stream="har_h2_l4", c=1.0),
    _r("har_h2_l4_c05", HAR, "stream", "H2-L4, c = 0,5", stream="har_h2_l4", c=0.5),
    _r("har_h2_l4m_c1", HAR, "stream", "H2-L4 на нормированных признаках Б-2", stream="har_h2_l4m", c=1.0),
    _r("har_h1_s", HAR, "stream", "H1 аддитивный, остаток к HAR (отн.), σ̂² = max(HAR(1+ĝ), ε)", stream="har_h1_s", c=1.0),
    _r("har_h1_raw", HAR, "stream", "H1 аддитивный сырой, σ̂² = max(HAR + ĝ, ε)", stream="har_h1_raw", c=1.0),
    _r("har_h3_w05", HAR, "h3", "H3 постоянный w = 0,5", h3="w05"),
    _r("har_h3_exp", HAR, "h3", "H3 w по всем закрытым (расширяющееся)", h3="exp"),
    _r("har_h3_roll252", HAR, "h3", "H3 w по последним 252 закрытым", h3="roll"),
    _r("har_h3_reg", HAR, "h3", "H3 w по группе режима {q0–q3}/{q4}", h3="reg"),
    _r("reg_r1a", REG, "stream", "R1 Б-2 + q_w", stream="reg_r1a", c=1.0),
    _r("reg_r1b", REG, "stream", "R1 Б-2 + q_w, стресс_RV, переход", stream="reg_r1b", c=1.0),
    _r("reg_r1c", REG, "stream", "R1 на форме остатка к HAR (L2)", stream="reg_r1c", c=1.0),
    _r("reg_r2a", REG, "experts", "R2 два режима {q0–q2}/{q3–q4}", stream="reg_r2a"),
    _r("reg_r2b", REG, "experts", "R2 три режима {q0–q1}/{q2–q3}/{q4}", stream="reg_r2b"),
    _r("reg_r3", REG, "r3", "R3 смесь экспертов R2a с логистикой на прошлом", stream="reg_r2a"),
    _r("reg_r4a", REG, "r4", "R4 HAR-J(б)/Б-2, порог θ_Y по прошлым годам", alt="b2"),
    _r("reg_r4b", REG, "r4", "R4 HAR-J(б)/лучшая H2, порог θ_Y", alt="h2best"),
    _r("reg_r4c", REG, "r4", "R4 стресс Г4 → Б-2, иначе HAR-J(б)", alt="b2", stress=True),
]
BY_ID = {r["row_id"]: r for r in ROWS}
H2_CAND = ("ctrl_b1", "har_h2_l2_c075", "har_h2_l2_c05", "har_h2_l4_c1", "har_h2_l4_c05", "har_h2_l4m_c1")
STREAMS_LOYO = ("ctrl_b2", "ctrl_b1", "obj_l1n", "obj_l1raw", "obj_l3x", "obj_l3l", "obj_l4", "obj_l5a", "obj_l5b", "obj_l5c",
                "har_h2_l4", "har_h2_l4m", "har_h1_s", "har_h1_raw", "reg_r1a", "reg_r1b", "reg_r1c")
STREAMS_BLOCK = STREAMS_LOYO


def trial_id(row: dict) -> str:
    cfg = {k: v for k, v in row.items() if k not in ("desc",)} | {"seed": SEED, "hp": "fix_i2_choice_g132", "schedule": "daily_expanding_I1"}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def trials() -> list:
    return [r for r in ROWS if r["trial"]]


def counts() -> dict:
    out = {OBJ: 0, HAR: 0, REG: 0}
    for r in trials():
        out[r["direction"]] += 1
    return out
