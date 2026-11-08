"""Этап I (G42–G44): запись `run_v4 --stage I` (прогнозы, таблицы, отчёт, реестр) и проверка `run_v4 --stage I --check` — ТОЛЬКО ЧТЕНИЕ (кэш, таблицы и отчёт должны уже лежать на диске; в памяти пересчитывается и сверяется)."""
from __future__ import annotations

from models_v2.s11 import v4_i_check as CK, v4_i_stage as ST


def main_i(check: bool) -> int:
    r = ST.stage_i(write=not check)
    if not check:
        print("стадия I записана; лучшая I %s; конфигураций %d" % (r["best"], len(ST.CFGS)))
        return 0
    bad = CK.all_checks(r)
    print("\n".join(bad) if bad else "G42–G44: все проверки пройдены")
    print("лучшая I: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
