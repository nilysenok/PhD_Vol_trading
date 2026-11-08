"""V5 лист FIN (Ж-226): стадия FIN — сборка таблиц G23 из СОХРАНЁННЫХ прогнозов, запись (`write`: таблицы, ключ FIN в CHECKPOINT, итоговый документ, REPRODUCTION), `--check` — ТОЛЬКО ЧТЕНИЕ (пересчёт в памяти, сверка с диском).
Конфигурации не расходуются (0), ALL_TRIALS не меняется."""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from models_v2.s12 import v5_0_io as IO, v5_fin_cost as CO, v5_fin_data as D, v5_fin_joint as JT, v5_fin_tabs as TB

CP = IO.REP / "CHECKPOINT.json"
NOTE = "итог программы V5, h = 1, 17 бумаг (M16 в выборе, SBER — диагностика)"
CLOSED = {"1 Graph-Score": "stop: ни одна GS не прошла три условия «go»; лучшая GS4; выигрыш к K8 держится на 2022 (числа — FIN_ALL.csv, раздел 14 итога)",
          "2 Spectral": "stop: лучшая SP1 прошла условия 1 и 2, не прошла 3; выигрыш держится на 2018 (числа — FIN_ALL.csv, FIN_YEARS.csv)",
          "3 Moments": "ABANDON: p Вальда K8 2017–2021 выше 0,01 — новой информации нет (DX1_MOMENTS.csv)",
          "4 Latent": "ABANDON: структура совпадает с GS3/GS4/GS5",
          "5 Jump": "ABANDON: значимые скачки несут долю QLIKE менее чем вдвое выше доли дней (DX1_JUMPS.csv)",
          "6 Break": "ABANDON: нет «go» направления 1 (план п. 3.6)",
          "7 сочетания": "ABANDON: нет «go» ни одного направления"}


def compute(measure: bool = False) -> dict:
    d = D.load()
    t = TB.tables_main(d)
    t.update({"FIN_JOINT": JT.joint(d), "FIN_ENCOMPASS": JT.encompass(d), "FIN_ABLATION": JT.ablation(d)})
    disk = None if measure else IO.read_csv("FIN_COST.csv")
    t["FIN_COST"] = CO.table(measure, disk)
    return dict(d=d, t=t)


def tables_sha() -> str:
    h = hashlib.sha256()
    for n in D.FIN_TABLES:
        h.update(n.encode() + IO.sha_file(IO.REP / (n + ".csv")).encode())
    return h.hexdigest()


def other_keys_sha(cp: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in cp.items() if k != "FIN"}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def fin_entry(r: dict) -> dict:
    cp = r["d"]["cp"]
    return {"configs_spent_by_FIN": 0, "used_total": cp["GS"]["used"] + cp["SP"]["used"], "budget": cp["budget"], "by_direction": {"GS": cp["GS"]["used"], "SP": cp["SP"]["used"]}, "go_any": False,
            "best_gs": r["d"]["best_gs"], "best_sp": r["d"]["best_sp"], "N_prior": cp["N_prior"], "N_declared_before_stats": cp["N_declared_before_stats"], "N_wide": cp["N_wide"], "closed_directions": CLOSED,
            "cache_keys": r["d"]["keys"], "tables_sha": tables_sha(), "other_keys_sha": other_keys_sha(cp), "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def write(r: dict) -> None:
    from models_v2.s12 import v5_fin_asm as B
    for name in D.FIN_TABLES:
        D.write_csv(r["t"][name], name, NOTE)
    cp = D.checkpoint()
    CP.write_text(json.dumps({**cp, "FIN": fin_entry(r)}, ensure_ascii=False, indent=1))
    B.write_docs()
