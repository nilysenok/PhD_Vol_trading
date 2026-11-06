"""Ж-212 (HS2, HS3, G6): реестр trials и CHECKPOINT.json. Первые 521 строка — побайтная копия hybrid_final_v1/ALL_TRIALS.csv;
новые строки: key `Ж-212:<направление>:<хэш конфигурации>`, program `Ж-212`. C без поля «архитектурное_отличие_от» не регистрируется.
Лимиты: B 12 · C 12 · D 12 · E 8 · F 6 · всего 50; превышение — ValueError. N = 521 + новые. Контроли (не trials) и решения HS11 — в CHECKPOINT.json."""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from models_v2.s8 import hs_paths as P

SRC = P.PRIOR_REP / "ALL_TRIALS.csv"
ALL = P.REP / "ALL_TRIALS.csv"
CHECK = P.REP / "CHECKPOINT.json"
LIMITS = {"har_features": 12, "residual_qlike": 12, "component": 12, "conditional": 8, "moe": 6}
TOTAL = 50
N_PRIOR = 521
PROGRAM = "Ж-212"
COLS = ["key", "program", "source", "direction", "config", "kind"]
ARCH = "архитектурное_отличие_от"


def tid(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def init() -> None:
    """Копия 521 строки побайтно (если файла нет); существующий реестр с новыми строками не трогается."""
    P.REP.mkdir(parents=True, exist_ok=True)
    if not ALL.exists():
        ALL.write_bytes(SRC.read_bytes())
    src, cur = SRC.read_bytes().splitlines(keepends=True), ALL.read_bytes().splitlines(keepends=True)
    if cur[:len(src)] != src or len(src) != N_PRIOR + 1:
        raise ValueError("первые %d строк реестра не равны hybrid_final_v1 побайтно" % (N_PRIOR + 1))


def load() -> pd.DataFrame:
    init()
    return pd.read_csv(ALL)


def used(df: pd.DataFrame) -> dict:
    n = df[df["program"] == PROGRAM]
    return {k: int((n["direction"] == k).sum()) for k in LIMITS}


def _state() -> dict:
    return json.loads(CHECK.read_text()) if CHECK.exists() else {"controls": [], "decisions": {}}


def write_check(df: pd.DataFrame, **extra) -> dict:
    u, st = used(df), _state()
    st.update({"limits": LIMITS, "limit_total": TOTAL, "used": u, "used_total": sum(u.values()), "N_prior": N_PRIOR,
               "N_final": N_PRIOR + sum(u.values()), "updated": time.strftime("%Y-%m-%d %H:%M"), **extra})
    CHECK.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float))
    return st


def register(direction: str, cfg: dict) -> str:
    """Новая уникальная конфигурация; повтор того же cfg — не trial (возвращает прежний ключ)."""
    if direction not in LIMITS:
        raise ValueError("чужое направление %s" % direction)
    if direction == "residual_qlike" and not str(cfg.get(ARCH, "")).strip():
        raise ValueError("C: нет поля «%s» (HS2)" % ARCH)
    df = load()
    key = "%s:%s:%s" % (PROGRAM, direction, tid(cfg))
    if key in set(df["key"]):
        return key
    u = used(df)
    if u[direction] + 1 > LIMITS[direction] or sum(u.values()) + 1 > TOTAL:
        raise ValueError("бюджет Ж-212 превышен: %s %s" % (direction, u))
    row = pd.DataFrame([(key, PROGRAM, "hybrid_struct_v1", direction, json.dumps(cfg, ensure_ascii=False, sort_keys=True), "trial")], columns=COLS)
    df = pd.concat([df, row], ignore_index=True)
    df.to_csv(ALL, index=False, lineterminator="\n")
    write_check(df)
    return key


def control(name: str, note: str) -> None:
    """Контроль (не trial): запись в CHECKPOINT.json."""
    st = _state()
    if name not in [c["name"] for c in st["controls"]]:
        st["controls"].append({"name": name, "note": note})
        CHECK.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float))


def decide(direction: str, decision: dict) -> None:
    """Решение HS11 по направлению (продолжать / стоп, причина, неизрасходованные trials)."""
    st = _state()
    st["decisions"][direction] = decision
    CHECK.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float))


def n_final() -> int:
    return N_PRIOR + sum(used(load()).values())
