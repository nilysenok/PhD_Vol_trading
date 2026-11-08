"""Лист K8-RQC (G35–G37): прогнозы K8, RQC1–RQC3 (`v4_rqc_run`, `v4_rqc_calib`, `v4_rqc_r3`), стадия, отчёт и проверки (вне путей прогноза). Вызывается из `run_v4 --stage RQC [--check]`."""
from __future__ import annotations

from models_v2.s11 import v4_io as IO, v4_rqc_check as CK, v4_rqc_report as RP, v4_rqc_stage as ST


def main_rqc(check: bool) -> int:
    r = ST.stage_rqc()
    (IO.REP / "RQC_REPORT.md").write_text(RP.build(r))
    if not check:
        print("стадия RQC записана; лучшая RQC %s" % r["best"])
        return 0
    bad = CK.all_checks(r)
    print("\n".join(bad) if bad else "G35–G37: все проверки пройдены")
    print("лучшая RQC: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
