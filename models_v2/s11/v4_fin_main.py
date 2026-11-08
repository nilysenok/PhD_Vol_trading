"""Итоговый лист FIN (G45–G47): `run_v4 --stage FIN` пишет таблицы и документы, `run_v4 --stage FIN --check` — ТОЛЬКО ЧТЕНИЕ (пересчёт в памяти и сверка с диском; ни один файл проекта не меняется)."""
from __future__ import annotations

from models_v2.s11 import v4_fin_check as CK, v4_fin_stage as ST


def main_fin(check: bool) -> int:
    r = ST.stage_fin(write=not check)
    if not check:
        print("стадия FIN записана; три лучшие %s; конфигураций расходовано 0" % r["top"])
        return 0
    bad = CK.all_checks(r)
    print("\n".join(bad) if bad else "G45–G47: все проверки пройдены")
    print("три лучшие: %s; crit_panel: %s" % (r["top"], {s: round(float(r["cs"][s]), 5) for s in r["top"] + ["k1", "k2", "k3", "d2"]}))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
