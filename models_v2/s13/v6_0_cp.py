"""V6-0, G8: CHECKPOINT.json, ключ `0`: перечень контролей «go», пороги условий 1–4 числами, годовые QLIKE лучшего из K2/K3 для условия 5, N = 722, N_wide = 3122, хэши входов, время записи. `ALL_TRIALS.csv` — только заголовок.
Запись — `write` (не в `--check`); при повторном запуске с теми же числами время записи сохраняется (идемпотентно)."""
from __future__ import annotations

import datetime as dt
import json

from models_v2.s13 import v6_0_io as IO

BUDGET = 18
CP = IO.REP / "CHECKPOINT.json"
ALL_TRIALS = IO.REP / "ALL_TRIALS.csv"
GO_RULE = ("пять условий одновременно (GATES_V6.md, «Правило go»): crit_SBER ≤ min(контролей) − δ; без целей 2017; без 2017–2018; без 1 % крайних дат; QLIKE лучше лучшего из K2/K3 не менее чем в 3 годах из 5; δ = 0,0005; "
           "min — по рядам класса «PIT-сопоставим» (список go_series)")


def counts() -> dict:
    v5 = json.loads((IO.V5REP / "CHECKPOINT.json").read_text())
    return {"N_V5_declared": v5["N_declared_before_stats"], "budget_V6": BUDGET, "N": v5["N_declared_before_stats"] + BUDGET, "N_wide_V5": v5["N_wide"], "N_wide": v5["N_wide"] + BUDGET}


def build(res_crit: dict, alt: dict) -> dict:
    th = res_crit["thresholds"]
    return {"program": "HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист 0", **counts(), "go_rule": GO_RULE, "go_series": th["go_series"], "thresholds": {k: v for k, v in th.items() if k != "go_series"},
            "noise": res_crit["noise"], "sensitivity_all_causal_prior": alt, "inputs_hash": IO.inputs_hash(), "code_hash": IO.code_hash(), "code_version": IO.CODE_VERSION, "used": 0,
            "note": "лист 0: конфигурации не расходуются; записано до первого прогноза V6 и до любого p", "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def stable(d: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in d.items() if k != "written_at"}))


def read() -> dict:
    return json.loads(CP.read_text()) if CP.exists() else {}


def write(entry: dict) -> None:
    old = read().get("0")
    if old and stable(old) == stable(entry):
        entry = {**entry, "written_at": old["written_at"]}
    CP.write_text(json.dumps({**read(), "0": entry}, ensure_ascii=False, indent=1))
    if not ALL_TRIALS.exists():
        ALL_TRIALS.write_text(IO.HEADER_TRIALS, encoding="utf-8")


def check(entry: dict) -> list:
    bad = []
    if not CP.exists() or "0" not in read():
        return ["G8: нет CHECKPOINT.json ключ 0 на диске"]
    if stable(read()["0"]) != stable(entry):
        bad.append("G8: CHECKPOINT ключ 0 на диске ≠ пересчёту")
    if (entry["N"], entry["N_wide"]) != (722, 3122):
        bad.append("G8: N = %s, N_wide = %s (ожидалось 722, 3122)" % (entry["N"], entry["N_wide"]))
    if not ALL_TRIALS.exists() or ALL_TRIALS.read_text(encoding="utf-8") != IO.HEADER_TRIALS:
        bad.append("G8: ALL_TRIALS.csv не только заголовок")
    return bad
