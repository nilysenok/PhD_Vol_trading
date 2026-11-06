"""Ж-214 (GATES_V2, V2-R2/R3): реестр trials и CHECKPOINT.json. Первые 541 строка — побайтная копия hybrid_struct_v1/ALL_TRIALS.csv
(521 Ж-204…Ж-210/211 + 20 Ж-212). Новые строки: key `Ж-214:<направление>:<хэш конфигурации>`, program `Ж-214`.
Лимиты (блок, п. 26): qlike_harj 8 · harq_j 12 · decomposition 16 · seasonality 8 · structural_ml 16 · market 10 · reconciliation 10 · всего 80.
qlike_harj и structural_ml без поля «отличие_от» (ключ реестра, который СУЩЕСТВУЕТ) не регистрируются. Повтор той же конфигурации — не trial.
N для всех поправок (Холм, DSR) = 541 + число новых trials."""
from __future__ import annotations

import json
import time

import pandas as pd

from models_v2.s8 import hs_ledger as HL
from models_v2.s9 import hv2_paths as P

SRC = P.PRIOR_REP / "ALL_TRIALS.csv"
ALL = P.REP / "ALL_TRIALS.csv"
CHECK = P.REP / "CHECKPOINT.json"
LIMITS = {"qlike_harj": 8, "harq_j": 12, "decomposition": 16, "seasonality": 8, "structural_ml": 16, "market": 10, "reconciliation": 10}
TOTAL = 80
N_PRIOR = 541
PROGRAM = "Ж-214"
COLS = ["key", "program", "source", "direction", "config", "kind"]
DIFF = "отличие_от"
NEED_DIFF = ("qlike_harj", "structural_ml")


def init(src=SRC, dst=ALL) -> None:
    """Копия 541 строки побайтно (если файла нет); существующий реестр с новыми строками не трогается."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    a, b = src.read_bytes().splitlines(keepends=True), dst.read_bytes().splitlines(keepends=True)
    if b[:len(a)] != a or len(a) != N_PRIOR + 1:
        raise ValueError("первые %d строк реестра не равны hybrid_struct_v1 побайтно" % (N_PRIOR + 1))


def load(dst=ALL) -> pd.DataFrame:
    init(dst=dst)
    return pd.read_csv(dst)


def used(df: pd.DataFrame) -> dict:
    n = df[df["program"] == PROGRAM]
    return {k: int((n["direction"] == k).sum()) for k in LIMITS}


def write_check(df: pd.DataFrame, path=CHECK, **extra) -> dict:
    u = used(df)
    st = json.loads(path.read_text()) if path.exists() else {"controls": [], "decisions": {}}
    st.update({"limits": LIMITS, "limit_total": TOTAL, "used": u, "used_total": sum(u.values()), "N_prior": N_PRIOR,
               "N_final": N_PRIOR + sum(u.values()), "updated": time.strftime("%Y-%m-%d %H:%M"), **extra})
    path.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float))
    return st


def register(direction: str, cfg: dict, dst=ALL, check=CHECK) -> str:
    """Записывает trial; ValueError при превышении лимита, неизвестном направлении, отсутствии/неверном «отличие_от». Повтор — тот же ключ."""
    if direction not in LIMITS:
        raise ValueError("неизвестное направление %r" % direction)
    df = load(dst)
    if direction in NEED_DIFF:
        ref = cfg.get(DIFF)
        if not ref or ref not in set(df["key"]):
            raise ValueError("%s: поле %r должно быть ключом реестра (получено %r)" % (direction, DIFF, ref))
    key = "%s:%s:%s" % (PROGRAM, direction, HL.tid(cfg))
    if key in set(df["key"]):
        return key
    u = used(df)
    if u[direction] >= LIMITS[direction] or sum(u.values()) >= TOTAL:
        raise ValueError("лимит trials: %s %d/%d, всего %d/%d" % (direction, u[direction], LIMITS[direction], sum(u.values()), TOTAL))
    row = pd.DataFrame([{"key": key, "program": PROGRAM, "source": "harj_hybrid_v2", "direction": direction,
                         "config": json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str), "kind": "trial"}], columns=COLS)
    row.to_csv(dst, mode="a", header=False, index=False, lineterminator="\n")
    write_check(load(dst), path=check)
    return key


def n_final(df: pd.DataFrame | None = None) -> int:
    return N_PRIOR + sum(used(load() if df is None else df).values())
