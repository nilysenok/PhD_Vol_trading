"""S-R-EVAL, документ (Ж-254 п. 16): разделы 7–9 отчёта `S_R_REPORT.md` — параметры, мощность, стоимость и воспроизводимость. Текст строится только из таблиц `S_R_*.csv` и ключа `S.R.eval`.
Публичное API: `sec7(T)`, `sec8(T)`, `sec9(T, ck)`."""
from __future__ import annotations

from models_v2.s18.r_doc_fmt import c_, dm_line, md, pv

R2 = ("S-RA2", "S-RCK")


def sec7(T: dict) -> str:
    p = T["params"]
    cnt = md(["Вариант", "обученных подгонок (L-BFGS-B)", "стенное время счёта, с"], [[v, int(pv(T, v, "fits", "n_fits")), c_(pv(T, v, "cost", "cpu_seconds"), 0)] for v in p["variant"].unique()])
    parts = []
    for v in R2:
        q = p[(p["variant"] == v) & (p["kind"] == "theta_q")]
        rows = [[n, *[c_(g[g["key"] == k]["value"].iloc[0], 4, True) for k in ("q01", "q05", "q50", "q95", "q99")]] for n, g in q.groupby("item", sort=False)]
        ln = p[(p["variant"] == v) & (p["kind"] == "lnratio")]
        s = ln[ln["item"] == "share_abs_gt_1e-3"]["value"].iloc[0]
        parts.append("### %s\n\nПараметры θ последних подгонок по месяцам с λ* > 0 (квантили по месяцам):\n\n" % v + md(["Параметр", "q01", "q05", "q50", "q95", "q99"], rows) + "\n\nln(C_R/cfg8) на активных датах: доля |ln| > 0,001 = %s; квантили %s." % (
            c_(s, 3), ", ".join("q%s = %s" % (r.key[1:], c_(r.value, 4, True)) for r in ln[ln["item"] == "q"].itertuples())))
    return ("## 7. Параметры\n\n" + dm_line(T) + " Число подгонок: активные месяцы × наборов (R, Rb — 10: 3 λ × 3 блока + финальная; Rc — 4) × 5 (seeds A2 / перезапуски C-K). Время — стенное, один процесс при четырёх параллельных (не CPU-часы).\n\n" + cnt +
            "\n\n" + "\n\n".join(parts))


def sec8(T: dict) -> str:
    rows = [[r.candidate, r.pair, r.window, int(r.n), int(r.lag), c_(r.sigma_lr, 4), c_(r.effect_pct, 4, True), c_(r.n_p05, 0), c_(r.n_p08, 0), c_(r.n_p05_hyp1pct, 0), c_(r.n_p08_hyp1pct, 0)] for r in T["power"].itertuples()]
    t = md(["Кандидат", "пара", "окно", "n", "лаг", "σ_LR", "эффект, %", "n для мощности 0,5", "n для 0,8", "n при −1 %: 0,5", "n при −1 %: 0,8"], rows)
    return ("## 8. Мощность\n\n" + dm_line(T) + " σ_LR разности потерь (NW-94), наблюдаемый эффект, n для мощности 0,5 и 0,8 при α = 0,05 односторонний; «—» — эффект не в пользу кандидата или нулевой; справа — гипотетический эффект −1 % QLIKE базы. "
            "Пары: М1 — X1_R против X1, М2 — F против W5-G, П — F против прежнего ряда. Недостаточная мощность не превращается в «улучшение невозможно».\n\n" + t)


def sec9(T: dict, ck: dict) -> str:
    return ("## 9. Стоимость и воспроизводимость\n\n" + dm_line(T) + " Все 6 вариантов записаны одним кодом `r_walk` после регистрации (реестр до прогноза, `S.R.registered`); seeds и перезапуски фиксированы; `r_walk --check` пересчитывает месяц каждого варианта побитно; "
            "`r_eval --check` и `r_doc --check` только читают (sha256 выходов до и после равны). Ключ `S.R.eval` записан %s до таблиц с p-значениями; Холм N = %d, N_wide = %d, m = %d / %d; бутстрап B = %d; лаг NW основной колонки = %d; K обрезки = %d. "
            "Число обученных подгонок и время счёта — в разделе 7. Это DEVELOPMENT / HISTORICAL WALK-FORWARD: результат исследовательский." % (ck["written_at"], ck["n_holm"], ck["n_holm_wide"], ck["m_family"], ck["m_proc"], ck["boot"], ck["lag_main"], ck["ktrim"]))
