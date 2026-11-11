"""S-R-EVAL, документ (Ж-254 п. 16): разделы 4–6 отчёта `S_R_REPORT.md` — концентрация (У6-R и срезы), защита и решения NEWCTRL-R / PROC, диагностика (θ ≠ 0, c, m, разложение, общее активное окно, годы). Текст строится только из таблиц `S_R_*.csv`.
Публичное API: `sec4(T)`, `sec5(T)`, `sec6(T)`."""
from __future__ import annotations

from models_v2.s18.r_doc_fmt import c_, cnc, dm_line, md, one, pv, skey, val, yn, yrs

CANDS, OLD = ("S-RA2", "S-RCK"), {"S-RA2": "S-A2", "S-RCK": "S-CK"}


def _pairs(c: str) -> list:
    return [("М1", "X1_R(%s)" % c, "X1"), ("М2", c, "W5-G"), ("П", c, OLD[c])]


def _ctab(T) -> str:
    rows = []
    for c in CANDS:
        for pid, h, b in _pairs(c):
            y1, y2, y3 = skey(T, h, b, "no_top_day"), skey(T, h, b, "var_share_top_day"), skey(T, h, b, "no_top_year")
            rows.append([c, pid, c_(cnc(T, h, b, "summary", "delta_total_pct"), 4, True), c_(cnc(T, h, b, "trim", "K=22"), 4, True), c_(cnc(T, h, b, "share_top1pct", "K=22"), 3), c_(cnc(T, h, b, "no_day", "2022-02-24"), 4, True),
                         c_(cnc(T, h, b, "no_year", "2022"), 4, True), c_(cnc(T, h, b, "no_top_day", y1), 4, True) + " (" + y1 + ")", c_(cnc(T, h, b, "no_top_year", y3), 4, True) + " (" + y3 + ")", c_(cnc(T, h, b, "var_share_top_day", y2), 3) + " (" + y2 + ")"])
    return md(["Кандидат", "пара", "Δ, %", "Δ_trim (22 + 22), %", "доля 1 % лучших дат (порог поручения 0,5)", "без 24.02.2022", "без 2022", "без самого влиятельного дня", "без самого влиятельного года", "доля дисперсии крупнейшего дня"], rows)


def _itab(T) -> str:
    k = T["conc"]
    k = k[k["item"].str.startswith("insample_")]
    rows = [[v, int(one(k, series=v, item="insample_months_theta_nonzero")["value"]), int(one(k, series=v, item="insample_months_theta_zero")["value"]), c_(one(k, series=v, item="insample_share_top1pct_median")["value"], 3),
             c_(one(k, series=v, item="insample_frac_rest99_better")["value"], 3)] for v in CANDS]
    return md(["Вариант", "месяцев θ ≠ 0", "месяцев θ = 0", "доля выигрыша обычной QLIKE от 1 % лучших пар пула (медиана по месяцам)", "доля месяцев, где остальные 99 % пар лучше нулевого старта"], rows)


def sec4(T: dict) -> str:
    return ("## 4. Концентрация выигрыша\n\n" + dm_line(T) + " Δ_trim — Δ в % QLIKE базы после удаления K = 22 дат с наименьшим и 22 дат с наибольшим d_t = l_кандидат − l_база (симметрично; база процентов — QLIKE базы на оставшихся 2154 датах); "
            "доля 1 % лучших дат — доля суммарного выигрыша от 22 лучших дат (определена при суммарном выигрыше > 0; прежнее условие поручения «≤ 0,5» заменено У6-R, см. отступление спецификации; на сохранённых рядах доля ≥ 1,0 у каждого принятого улучшения проекта). "
            "Срезы — только диагностика, не новая выборка; основная оценка — все 2198 дат.\n\n" + dm_line(T) + "\n\n" + _ctab(T) +
            "\n\nВ выборке по месяцам (S-C-WHY разд. 5.4; параметры — средние θ последней подгонки месяца на закрытом пуле, у A2 — по среднему θ seeds):\n\n" + dm_line(T) + "\n\n" + _itab(T))


def _grow(r) -> list:
    return [r.candidate, r.pair, r.id, r.condition, val(r.value), val(r.threshold), yn(r.passed)]


def sec5(T: dict) -> str:
    g, p = T["guard"], T["proc"]
    body = md(["Кандидат", "пара", "условие", "описание", "значение", "порог", "выполнено"], [_grow(r) for r in g.itertuples() if r.id != "решение"])
    d = g[g["id"] == "решение"]["value"].iloc[0]
    pr = "; ".join("%s: %s" % (r.pid, ("да" + (", подтверждено" if r.confirmed else ", не подтверждено статистически")) if r.passed else "нет") for r in p.itertuples())
    return ("## 5. Защита и решения\n\n" + dm_line(T) + " Пороги У1-S…У3-S — этапа S без изменений (0,415 / 0,50 / 1,00 %); У5-S — Δ без лучшего для F_R периода ≤ 0; У6-R — Δ_trim < 0 для F против W5-G и X1_R против X1. Победа в каждом году не требуется.\n\n" + body +
            "\n\nРешение NEWCTRL-R (порядок S-RA2 → S-RCK, значимость не требуется, выбор по минимуму QLIKE запрещён): **" + d + "**\n\nРешение PROC (процедура против прежних рядов той же архитектуры): **" + pr + "**\n\n"
            "Независимое подтверждение: нет (процедура выбрана после разбора S-C-WHY на раскрытой истории; подтверждение — только на данных после 10.11.2026 на замороженном коде, процедура Л5–Л6).")


def _ytab(T, cand) -> str:
    rows = []
    for p in T["years"]["period"].unique():
        r1, r2 = yrs(T, "X1_R(%s)" % cand, "X1", p), yrs(T, cand, "W5-G", p)
        rows.append([p, int(r1["n"]), c_(r1["delta_pct"], 4, True), c_(r1["contrib_pct"], 4, True), c_(r2["delta_pct"], 4, True), c_(r2["contrib_pct"], 4, True)])
    return md(["Период (t⁺)", "n", "Δ М1, %", "вклад М1, п. п.", "Δ М2, %", "вклад М2, п. п."], rows)


def sec6(T: dict) -> str:
    ab = T["abl"]
    a = md(["Кандидат", "вид", "показатель", "окно", "n", "Δ, %"], [[r.candidate, r.kind, r.item, r.window, int(r.n), c_(r.value, 4, True)] for r in ab.itertuples()])
    th = md(["Вариант", "активных месяцев", "месяцев θ = 0", "доля θ = 0", "λ* = 0,1", "λ* = 0,01", "λ* = 0,001", "c (медиана)", "m (медиана)", "κ = c/m (медиана)"],
            [[v, int(pv(T, v, "months", "active")), int(pv(T, v, "months", "theta_zero")), c_(pv(T, v, "months", "share_theta_zero"), 3)] + [int(pv(T, v, "lam", k)) for k in ("0.1", "0.01", "0.001")] +
             ([c_(pv(T, v, "clip", "c_median"), 3), c_(pv(T, v, "clip", "m_median"), 4), c_(pv(T, v, "clip", "kappa_median"), 3)] if v in ("S-RA2", "S-RCK", "S-RA2c", "S-RCKc") else ["∞ (без обрезки)", "1", "—"]) for v in T["params"]["variant"].unique()])
    return ("## 6. Диагностика\n\n" + dm_line(T) + " Доля месяцев с θ ≠ 0 и λ*, c, m, κ — по месяцам переоценки (полная таблица выбора, w_λ и G по блокам — `S_R_SEL.csv`).\n\n" + th +
            "\n\nРазложение Δ(F_R против W5-G) на «эффект при замороженном π» (F_fr = B_mix^(1−π)·C_R^π·e^{βg} с сохранёнными π, β) и «реакцию слоя» (F_R − F_fr); Δ на общем активном окне (по датам прогноза d; компенсирует включение с 630 пар):\n\n" +
            dm_line(T) + "\n\n" + a + "\n\n### S-RA2\n\n" + _ytab(T, "S-RA2") + "\n\n### S-RCK\n\n" + _ytab(T, "S-RCK") + "\n\nВклады каждого периода для всех пар — `S_R_YEARS.csv`, срезы — `S_R_CONC.csv`.")
