"""V6 лист Z3: стадия — сборка семи таблиц из СОХРАНЁННЫХ прогнозов (`compute`), запись (`write`: таблицы, документ, CHECKPOINT_WHY.json), хэш выходов. `--check` (v6_why_check) только читает.
Конфигурации не расходуются (0); ключи CHECKPOINT.json V6 и ALL_TRIALS.csv не трогаются."""
from __future__ import annotations

import datetime as dt
import json
import time

from models_v2.s13 import v6_0_io as IO, v6_why_a as A, v6_why_b as B, v6_why_data as DT


def tables(d: dict) -> dict:
    return {"WHY_YEARS": A.years_table(d), "WHY_TAILS": A.tails_table(d), "WHY_REGIME": A.regime_table(d), "WHY_MZ": B.mz_table(d), "WHY_CAPTURE": B.capture_table(d), "WHY_ROLL": B.roll_table(d), "WHY_GSIZE": B.gsize_table(d)}


def compute() -> dict:
    t0 = time.time()
    d = DT.load()
    return dict(d=d, t=tables(d), seconds=time.time() - t0)


def own_digest() -> dict:
    """SHA16 всех собственных выходов (таблицы, ключ, документ): для «до и после» у `--check`."""
    names = [DT.WHY / (n + ".csv") for n in DT.TABLES] + [DT.CP_FILE, DT.DOC]
    return {p.name: (IO.sha_file(p)[:16] if p.exists() else "нет") for p in names}


def entry(n_dates: int, old: dict | None) -> dict:
    sh = own_digest()
    e = {"program": "HAR-J Hybrid V6 (Ж-230), SBER, h=1, лист Z3 (причины)", "configs_spent": 0, "n_dates": n_dates, "tables": DT.TABLES, "tables_sha": {n: sh[n + ".csv"] for n in DT.TABLES}, "doc_sha": sh[DT.DOC.name],
         "inputs_hash": IO.inputs_hash(), "code_hash_v6_why": DT.code_hash(), "code_version": DT.CODE_VERSION, "note": "диагностика, не отбор; потолок, не OOS"}
    stable = lambda x: {k: v for k, v in x.items() if k != "written_at"}
    e["written_at"] = old["written_at"] if old and stable(old) == stable(e) else dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return e


def write(r: dict) -> None:
    from models_v2.s13 import v6_why_doc as W0, v6_why_doc2 as W
    for n in DT.TABLES:
        DT.write_csv(r["t"][n], n)
    DT.DOC.write_text(W.build(W0.load_tables()), encoding="utf-8")
    old = json.loads(DT.CP_FILE.read_text()) if DT.CP_FILE.exists() else None
    DT.CP_FILE.write_text(json.dumps(entry(len(r["d"]["dates"]), old), ensure_ascii=False, indent=1), encoding="utf-8")
