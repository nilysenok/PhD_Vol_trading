"""Этап H (G38–G41): запись `run_v4 --stage H` (прогнозы, таблицы, отчёт, реестр) и проверка `run_v4 --stage H --check` — ТОЛЬКО ЧТЕНИЕ (кэш, таблицы и отчёт должны уже лежать на диске; в памяти пересчитывается и сверяется)."""
from __future__ import annotations

from models_v2.s11 import v4_h_stage as ST, v4_hd_check as CK


def main_h(check: bool) -> int:
    r = ST.stage_h(write=not check)
    if not check:
        print("стадия H записана; лучшая H %s; конфигураций %d" % (r["best"], len(ST.CFGS)))
        return 0
    bad = CK.all_checks(r)
    print("\n".join(bad) if bad else "G39–G41: все проверки пройдены")
    print("лучшая H: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
