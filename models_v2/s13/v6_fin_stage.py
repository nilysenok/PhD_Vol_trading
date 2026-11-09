"""V6 лист FIN (Ж-227): стадия FIN — сборка таблиц из СОХРАНЁННЫХ прогнозов, запись (`write`: таблицы, ключ FIN в CHECKPOINT, итоговый документ, REPRODUCTION), `--check` — ТОЛЬКО ЧТЕНИЕ (пересчёт в памяти, сверка с диском).
Конфигурации не расходуются (0), `ALL_TRIALS.csv` и ключи 0…6 CHECKPOINT не меняются."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_stage as S1, v6_fin_data as D, v6_fin_tabs as TB


def compute() -> dict:
    t0 = time.time()
    d = D.load()
    return dict(d=d, t=TB.tables(d), seconds=time.time() - t0)


def tables_sha() -> dict:
    return {n: IO.sha_file(IO.REP / (n + ".csv"))[:16] for n in D.TABLES}


def other_keys_sha(cp: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in cp.items() if k != "FIN"}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def fin_entry(cp: dict, n_dates: int) -> dict:
    """Ключ FIN: без времени счёта в содержании (стабилен), хэши всех выходов таблиц и прочих ключей."""
    return {"program": "HAR-J Hybrid V6 (Ж-227), SBER, h=1, лист FIN", "configs_spent_by_FIN": 0, "used_total": cp["6"]["used"], "budget": cp["6"]["budget"], "N": cp["0"]["N"], "N_wide": cp["0"]["N_wide"],
            "n_dates": n_dates, "series": len(D.SER), "tables": D.TABLES, "tables_sha": tables_sha(), "other_keys_sha": other_keys_sha(cp), "inputs_hash": IO.inputs_hash(), "code_hash_v6_fin": D.code_hash(),
            "code_version": D.CODE_VERSION, "note": "лист FIN: потолок, не OOS", "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def write(r: dict) -> None:
    from models_v2.s13 import v6_fin_asm as A
    for name in D.TABLES:
        D.write_csv(r["t"][name], name)
    cp = CP0.read()
    entry = fin_entry(cp, len(r["d"]["w"]))
    old = cp.get("FIN")
    if old and S1.stable(old) == S1.stable(entry):
        entry["written_at"] = old["written_at"]
    CP0.CP.write_text(json.dumps({**cp, "FIN": entry}, ensure_ascii=False, indent=1))
    A.write_docs()
