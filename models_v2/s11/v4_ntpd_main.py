"""Лист NTP (G29–G31): прогнозы NTP1–NTP4 (`v4_ntp*`), состояния/отчёт/проверки (`v4_ntpd_*`, вне путей прогноза). Вызывается из `run_v4 --stage NTP [--check]`."""
from __future__ import annotations

from models_v2.s11 import v4_ntp_stage as ST, v4_ntpd_check as CK, v4_ntpd_report as RP, v4_ntpd_states as STS
from models_v2.s11 import v4_io as IO


def main_ntp(check: bool) -> int:
    r = ST.stage_ntp()
    st, pb = STS.states(r), STS.pi_behavior(r)
    IO.write_csv(st, "NTP_STATES.csv", "доказательство по состояниям (Ж-225 п. 4): группы по PIT-доверию NTP3 на дату прогноза (π > 0,5 / ≤ 0,5), D = QLIKE(C8 или K1) − QLIKE(M), M = √(K2·K3), M16; DEVELOPMENT / HISTORICAL WALK-FORWARD; " + r["note"])
    (IO.REP / "NTP_REPORT.md").write_text(RP.build(r, st, pb))
    if not check:
        print("стадия NTP записана; лучшая NTP %s" % r["best"])
        return 0
    bad = CK.all_checks(r, st, pb)
    print("\n".join(bad) if bad else "G29–G31: все проверки пройдены")
    print("лучшая NTP: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
