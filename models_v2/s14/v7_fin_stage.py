"""V7 лист FIN (Ж-232 п. 16): стадия FIN — сборка таблиц из СОХРАНЁННЫХ прогнозов, запись (`write`: таблицы, ключ FIN в CHECKPOINT, итоговый документ, REPRODUCTION, PNG), `--check` — ТОЛЬКО ЧТЕНИЕ
(пересчёт в памяти, сверка с диском). Конфигурации не расходуются (0), `ALL_TRIALS.csv` и ключи 0…DX CHECKPOINT не меняются. Запуск: `python3 -W ignore -m models_v2.s14.run_v7 --stage FIN [--check]`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time

from models_v2.s14 import v7_0_io as IO, v7_fin_asm as A, v7_fin_data as D, v7_fin_tabs as TB


def compute() -> dict:
    t0 = time.time()
    d = TB.load()
    return dict(d=d, t=TB.all_tables(d), seconds=time.time() - t0)


def tables_sha() -> dict:
    return {n: IO.sha_file(IO.REP / (n + ".csv"))[:16] for n in D.TABLES}


def other_keys_sha(cp: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in cp.items() if k != "FIN"}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def stable(e: dict) -> dict:
    return {k: v for k, v in e.items() if k != "written_at"}


def fin_entry(cp: dict, n_dates: int) -> dict:
    """Ключ FIN: без времени счёта в содержании (стабилен); хэши таблиц, документа и прочих ключей."""
    return {"program": "HAR-J Hybrid V7 (Ж-232), SBER, h=1, лист FIN", "configs_spent_by_FIN": 0, "places": 11, "computed": len(D.CFG), "not_run_by_condition": [9, 11], "reserve_used": False, "n_dates": n_dates, "series": len(D.SER),
            "tables": D.TABLES, "tables_sha": tables_sha(), "doc_sha": IO.sha_file(A.FINAL)[:16] if A.FINAL.exists() else "", "other_keys_sha": other_keys_sha(cp), "inputs_hash": IO.inputs_hash(), "code_hash_v7_fin": D.code_hash(),
            "code_version": D.CODE_VERSION, "note": "лист FIN: потолок, не OOS", "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def write(r: dict) -> None:
    IO.REP.mkdir(parents=True, exist_ok=True)
    for n in D.TABLES:
        D.write_csv(r["t"][n], n)
    A.write_docs(A.load())
    cp = D.read_cp()
    entry = fin_entry(cp, len(r["d"]["rv"]))
    old = cp.get("FIN")
    if old and stable(old) == stable(entry):
        entry["written_at"] = old["written_at"]
    IO.merge_checkpoint("FIN", entry)


def run(check: bool) -> int:
    from models_v2.s14 import v7_fin_check as CK
    before = IO.outputs_digest()
    r = compute()
    if not check:
        write(r)
        print("FIN записан: %d таблиц, документ %s, %.0f с" % (len(D.TABLES), A.FINAL.name, r["seconds"]))
        return 0
    bad = CK.all_bad(r)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: FIN — таблицы, документ (%d чисел), реестр и CHECKPOINT сошлись; выходы не изменены (хэш %s)" % (CK.n_numbers(), after[:16]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run("--check" in sys.argv))
