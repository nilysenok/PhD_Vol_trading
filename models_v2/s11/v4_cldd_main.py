"""Лист CLD (G32–G34): прогнозы CLD1–CLD4 (`v4_cld_*`), предсказуемость D/отчёт/проверки (`v4_cldd_*`, вне путей прогноза). Вызывается из `run_v4 --stage CLD [--check]`."""
from __future__ import annotations

from models_v2.s11 import v4_cld_stage as ST, v4_cldd_check as CK, v4_cldd_pred as PR, v4_cldd_report as RP, v4_io as IO


def main_cld(check: bool) -> int:
    r = ST.stage_cld()
    pred = PR.pred_table(r["fc"])
    IO.write_csv(pred, "CLD_PRED.csv", "предсказуемость D = QLIKE(H) − QLIKE(M) вне выборки (Ж-225 п. 5): D̂ по закрытым парам до начала месяца; M16; окна и месяцы цели; R² — к историческому среднему D окна оценки; "
                 "знак — доля совпадений знака D̂ и D (share_pos — доля D>0); DEVELOPMENT / HISTORICAL WALK-FORWARD; " + r["note"])
    (IO.REP / "CLD_REPORT.md").write_text(RP.build(r, pred))
    if not check:
        print("стадия CLD записана; лучшая CLD %s" % r["best"])
        return 0
    bad = CK.all_checks(r, pred)
    print("\n".join(bad) if bad else "G32–G34: все проверки пройдены")
    print("лучшая CLD: %s; crit_panel: %s" % (r["best"], r["crit"].set_index("series")["crit_panel"].round(5).to_dict()))
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
