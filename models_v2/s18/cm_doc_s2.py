"""S-C-EVAL, документ (Ж-254 п. 16; `S_C_SPEC.md`, разд. 4.2–4.6): разделы 4–6 отчёта — структурные сравнения К1–К3 и абляция C-KA, защита и решения NEWCTRL / STRUCT, распределение ln 𝓜. Текст строится только из таблиц `S_C_*.csv`.
Публичное API: `sec4(T)` — К1 (S-CK против S-CD), К2 (против S-C7), К3 (против S-CKL), S-CKA − S-CK (F и слой X1_C): Δ, интервалы, p, Холм m = 3, SPA; `sec5(T)` — строки защиты по кандидатам, пять уровней (техническая корректность /
численное улучшение / ограничения / статистическая подтверждённость / независимое подтверждение), решения из GUARD; `sec6(T)` — ln 𝓜 на D_M, по годам, доли компонент против ŝ."""
from __future__ import annotations

from models_v2.s18.cm_doc_fmt import c_, confirmed, dm_line, md, one, stat, val, word, yn, yrs

THR = {"У1-S": "0,415", "У2-S": "0,50", "У3-S": "1,00"}
SI = (("К1", "S-CK", "S-CD", "форма: компонентная против прямой на той же информации"), ("К2", "S-CK", "S-C7", "обучение внутри гибрида против раздельных HAR (K7-класс, тот же фактор)"),
      ("К3", "S-CK", "S-CKL", "привязка разложения ко времени (сдвиг на 252 сессии)"))


def _row(s, lab, what) -> list:
    return ["%s: %s" % (lab, what), c_(s["qlike_h"], 6), c_(s["qlike_b"], 6), c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_dm"], 4), word(s["delta_pct"])]


def _ytab(T) -> str:
    rows = []
    for p in T["years"]["period"].unique():
        cells = [p, int(yrs(T, "S-CK", "S-CD", p)["n"])]
        for _, c, b, _w in SI:
            r = yrs(T, c, b, p)
            cells += [c_(r["delta_pct"], 4, True), c_(r["contrib_pct"], 4, True)]
        rows.append(cells)
    return md(["Период (t⁺)", "n", "Δ К1, %", "вклад К1", "Δ К2, %", "вклад К2", "Δ К3, %", "вклад К3"], rows)


def sec4(T: dict) -> str:
    rows = [_row(stat(T, i, "struct"), i, w) for i, _c, _b, w in SI]
    for lab in ("F", "слой X1_C"):
        rows.append(_row(stat(T, "S-CKA|S-CK %s" % lab, "ablation"), "S-CKA − S-CK (%s)" % lab, "дополнительная компонентная задача"))
    hold = md(["Сравнение", "Δ, %", "p DM NW-8", "Холм m = 3", "Холм N = 836", "p SPA (c, один кандидат)", "вывод"],
              [[i, c_(stat(T, i, "struct")["delta_pct"], 4, True), c_(stat(T, i, "struct")["p_dm"], 4), c_(stat(T, i, "struct")["p_holm_m"], 4), c_(stat(T, i, "struct")["p_holm_N"], 4), c_(stat(T, i, "struct")["p_spa_c"], 4),
                stat(T, i, "struct")["verdict"]] for i, _c, _b, _w in SI])
    g = T["guard"]
    return ("## 4. Структурные сравнения\n\n" + dm_line(T) + " Δ — % QLIKE второго ряда (базы пары); все сравнения на уровне F (с W5-слоем), n = 2198, p DM — NW-8, односторонний «первый лучше». К1 — тот же набор входов и то же число параметров (6); К2, К3 — контроли. "
            "Строки S-CKA − S-CK вне семейств, p без поправки.\n\n" + md(["Пара", "QLIKE первого", "QLIKE второго", "Δ, %", "90 % интервал", "p DM", "первый"], rows) + "\n\nХолм по структуре (m = 3, N программы) и SPA:\n\n" + dm_line(T) + "\n\n" + hold +
            "\n\n«Компонентная структура дала численный эффект» — только при К1 ∧ К2 ∧ К3: **" + g[g["id"] == "решение STRUCT"]["value"].iloc[0] + "**.\n\nВклады периодов (К1–К3), п. п. QLIKE базы:\n\n" + dm_line(T) + "\n\n" + _ytab(T))


def _guard(T, cand) -> str:
    g = T["guard"]
    g = g[(g["candidate"] == cand) & (g["id"] != "кандидат")]
    return md(["Условие", "пара", "значение", "порог", "выполнено"], [[r.id, r.pair, val(r.value), val(r.threshold), yn(r.passed)] for r in g.itertuples()])


def sec5(T: dict) -> str:
    g, sp = T["guard"], T["stats"]
    kand = {c: g[(g["candidate"] == c) & (g["id"] == "кандидат")]["value"].iloc[0] for c in ("S-CK", "S-CKA")}
    fam = sp[sp["kind"].isin(["main", "struct"])]
    s = {k: stat(T, k) for k in ("S-CK|X1", "S-CK|W5-G", "S-CKA|X1", "S-CKA|W5-G")}
    st = g[g["candidate"] == "структура"]
    return ("## 5. Защита и решения NEWCTRL / STRUCT\n\n" + dm_line(T) + " Пороги — константы спецификации (У1-S %s %%, У2-S %s %%, У3-S %s %%); применяются к обеим парам: М1 (X1_C против X1) и М2 (F против W5-G); h1 — 2018–2021, h2 — 2022–2026. "
            "У4-C: QLIKE кандидата ниже S-CKL и S-C7. Порядок S-CK → S-CKA; S-CD, S-C7, S-CKL кандидатами не являются.\n\n" % (THR["У1-S"], THR["У2-S"], THR["У3-S"]) +
            "### S-CK\n\n" + _guard(T, "S-CK") + "\n\n### S-CKA\n\n" + _guard(T, "S-CKA") + "\n\nСтруктура:\n\n" + md(["Условие", "значение", "выполнено"], [[r.id, val(r.value), yn(r.passed)] for r in st.itertuples()]) +
            "\n\nРешение NEWCTRL (выведено кодом из строк выше): **" + g[g["id"] == "решение"]["value"].iloc[0] + "**.\n\nРешение STRUCT: **" + g[g["id"] == "решение STRUCT"]["value"].iloc[0] + "**.\n\n"
            "Раздельно:\n1. Техническая корректность — гейты листов IMPL и E.G1–E.G3 в `GATES_S.md` (отступления — там).\n"
            "2. Численное улучшение: S-CK М1 Δ = %s %%, М2 Δ = %s %%; S-CKA М1 Δ = %s %%, М2 Δ = %s %%.\n" % (c_(s["S-CK|X1"]["delta_pct"], 4, True), c_(s["S-CK|W5-G"]["delta_pct"], 4, True), c_(s["S-CKA|X1"]["delta_pct"], 4, True), c_(s["S-CKA|W5-G"]["delta_pct"], 4, True)) +
            "3. Прохождение зарегистрированных ограничений: S-CK — %s, S-CKA — %s (число выполненных из 10).\n" % (kand["S-CK"], kand["S-CKA"]) +
            "4. Статистическая подтверждённость: подтверждённых гипотез из 11 (8 кандидаты × базы и 3 структуры) — %d (условие: Δ < 0, p Холма N = 836 < 0,05 и p SPA < 0,05); наименьший p Холма (N = 836) = %s, наименьший p SPA (c) = %s.\n" %
            (confirmed(T), c_(fam["p_holm_N"].min(), 4), c_(fam["p_spa_c"].min(), 4)) +
            "5. Независимое подтверждение: нет (только данные после 10.11.2026 на замороженном коде, процедура Л5–Л6).\n\n" + "Справочно, п. 4.6: " + g[g["id"] == "S-CD справочно"]["value"].iloc[0] + ".")


def sec6(T: dict) -> str:
    L = T["lnm"]
    d = L[L["kind"] == "dist"]
    rows = [[r.variant, r.key, int(r.n), c_(r.share_abs_gt_1e3, 3), c_(r.mean, 5, True), c_(r.std, 5), c_(r.min, 4, True), c_(r.q01, 4, True), c_(r.q50, 5, True), c_(r.q99, 4, True), c_(r.max, 4, True)] for r in d.itertuples()]
    y = L[L["kind"] == "year"]
    vs, yrs_ = ("S-CD", "S-CK", "S-CKA", "S-CKL", "S-C7"), sorted(y["key"].unique())
    yr = [[k] + [c_(one(y, variant=v, key=k)["mean"], 5, True) + " / " + c_(one(y, variant=v, key=k)["share_abs_gt_1e3"], 2) for v in vs] for k in yrs_]
    sh = L[L["kind"] == "share"]
    sr = [[k] + [c_(one(sh, variant=v, key=k)[col], 3) for v in ("S-CK", "S-CKA") for col in ("fc_over_c", "fp_over_c")] + [c_(one(sh, variant="S-CK", key=k)["s_c"], 3), c_(one(sh, variant="S-CK", key=k)["s_p"], 3)] for k in yrs_]
    return ("## 6. Распределение ln 𝓜 (меняется ли общий уровень)\n\n" + dm_line(T) + " 𝓜 = C_V/cfg8; при нулевых параметрах 𝓜 ≡ 1. Колонка «доля» — доля дат с |ln 𝓜| > 1e-3. «активные» — даты с 2018-07-02 (до включения 𝓜 = 1 по протоколу).\n\n" +
            md(["Вариант", "даты", "n", "доля дат с модулем ln 𝓜 > 1e-3", "среднее", "σ", "min", "q01", "медиана", "q99", "max"], rows) + "\n\nПо годам t⁺ (среднее ln 𝓜 / доля |ln 𝓜| > 1e-3):\n\n" + dm_line(T) + "\n\n" + md(["Год"] + list(vs), yr) +
            "\n\nСредние доли компонент в прогнозе по годам против долей ŝ (S-CK и S-CKA: F_c/C, F_p/C; ŝ_c, ŝ_p — доли окна 22 сессий, общие для всех вариантов):\n\n" + dm_line(T) + "\n\n" +
            md(["Год", "CK F_c/C", "CK F_p/C", "CKA F_c/C", "CKA F_p/C", "ŝ_c", "ŝ_p"], sr))
