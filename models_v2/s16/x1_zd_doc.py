"""ZD: отчёт `ZD_REPORT.md` ГЕНЕРАТОРОМ из таблиц (ни одно число не вписано руками; `--check` сверяет файл с `build`). Файлов не пишет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_zd_tab as TB

FIRST = "диагностика, не отбор; историческая; без конфигураций"
f4 = lambda v: "н/д" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.4f" % v     # noqa: E731


def md(rows: list, head: list) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def states_md(st: pd.DataFrame, y: str, periods: list) -> str:
    g, rows = st[st["y"] == y], []
    for p in periods:
        a = {s: g[(g["state"] == s) & (g["period"] == p)].iloc[0] for s in ("E", "N", "all")}
        rows.append([p, int(a["E"]["n"]), f4(a["E"]["mean_D"]), int(a["N"]["n"]), f4(a["N"]["mean_D"]), f4(a["all"]["mean_D"]), f4(a["all"]["sum_D"])])
    return md(rows, ["период (t⁺)", "n E", "D E", "n N", "D N", "D все", "Σ D"])


def auc_md(pr: pd.DataFrame, y: str) -> str:
    g = pr[pr["y"] == y]
    return md([[r.period, r.n, f4(r.share_neg), f4(r.auc_a), f4(r.auc_c), f4(r.auc_g), f4(r.rho_b)] for r in g.itertuples()], ["период", "n", "доля D<0", "AUC (а) z", "AUC (в) e_d", "AUC (г) сдвиг 252", "ρ (б)"])


def verdict(pr: pd.DataFrame, y: str) -> str:
    """Механический вывод из таблицы: AUC за весь период и по годам 2018–2025."""
    g = pr[pr["y"] == y].set_index("period")
    w, yrs = g.loc["all"], g.loc[[str(v) for v in range(2018, 2026)]]
    ok = yrs["auc_a"].dropna()
    return ("%s: за весь период AUC (а) = %s, (в) только e_d = %s, (г) сдвиг 252 = %s; (а) выше (в): %s; (а) выше (г): %s; AUC (а) > 0,5 в %d из %d лет 2018–2025; ρ (б) = %s."
            % (y, f4(w["auc_a"]), f4(w["auc_c"]), f4(w["auc_g"]), "да" if w["auc_a"] > w["auc_c"] else "нет", "да" if w["auc_a"] > w["auc_g"] else "нет", int((ok > 0.5).sum()), len(ok), f4(w["rho_b"])))


def half_md(pr: pd.DataFrame) -> str:
    g = pr[pr["y"] == "Y2"]
    return md([[r.period, f4(r.D_hi), f4(r.D_lo)] for r in g.itertuples()], ["период", "D в верхней половине прогноза (б)", "D в нижней половине"])


def build(t: dict, num: dict) -> str:
    st, pr, ca, cs = t["STATES"], t["PRED"], t["CALIB"], t["CAUSES"]
    yrs = TB.PERIODS
    out = [FIRST, "", "# ZD: предсказуема ли полезность поправки заранее (Ж-246; SBER, h = 1)", "",
           "D = l(Y) − l(X1), l = RV/F − ln(RV/F) − 1; D < 0 — поправка помогла. Даты общие, n = %d (d %s … %s). Прогнозов RV, конфигураций и строк реестра нет; код %s." % (num["n_dates"], num["first"], num["last"], num["code"]),
           "Тождество Σ D по состояниям = по годам = всего: %s." % ("выполнено" if not num["identity_bad"] else "НАРУШЕНО: " + "; ".join(num["identity_bad"])), ""]
    for y in ("Y2", "Y1", "Y3"):
        out += ["## D по состояниям E / N × период, %s" % y, "", states_md(st, y, yrs), ""]
    out += ["## Предсказуемость (с 2018-07-02, пул — пары с t⁺ ≤ D)", ""]
    for y in ("Y2", "Y1", "Y3"):
        out += ["### " + y, "", auc_md(pr, y), ""]
    out += ["## Вывод по таблицам (механически)", ""] + ["- " + verdict(pr, y) for y in ("Y2", "Y1", "Y3")]
    out += ["", "## Y2: средний D в половинах прогноза (б) — описание, не модель и не выбор эксперта", "", half_md(pr), "",
            "## Калибровка (а), Y2, весь период после включения", "", md([[int(r.decile), int(r.n), f4(r.mean_pred), f4(r.share_neg)] for r in ca.itertuples()], ["дециль", "n", "средний прогноз", "доля D<0"]), "",
            "## Причины 1–6 плана Z: что добавила ZD", "", md([[int(r.cause), r.item, r.period, f4(r.value), r.source] for r in cs.itertuples()], ["причина", "показатель", "период", "значение", "источник"]), "",
            "Оговорки: история 2018–2026 многократно просмотрена; контроль (г) — циклический сдвиг по всему ряду (у первых 252 дат признаки берутся с конца ряда — это контроль, не прогноз). Не отбор, без выбора эксперта.", ""]
    return "\n".join(out)
