"""S-A-EVAL, документ (Ж-254 п. 16; `S_A_SPEC.md`, разд. 6.5, 8): разделы 4–6 отчёта — абляции, защита У1-S…У5-S и решение, концентрация. Текст строится только из таблиц `S_A_*.csv`.
Публичное API: `sec4(T)` — абляции A0 → A1 → A2 → A2p (F и слой X1_A) с интервалами и p DM (справочно, вне семейства); `sec5(T)` — строки защиты по кандидатам, пять уровней (техническая корректность / численное улучшение / ограничения /
статистическая подтверждённость / независимое подтверждение), решение из GUARD; `sec6(T)` — концентрация и чувствительность (диагностика, не условие)."""
from __future__ import annotations

from models_v2.s18.a_doc_fmt import c_, cnc, dm_line, md, stat, val, yn

NAME = {"F": "F (с W5-слоем)", "слой X1_A": "слой X1_A"}
ABL = (("S-A1", "S-A0", "структура междневной памяти (a)"), ("S-A2", "S-A1", "внутридневные веса (b)"), ("S-A2", "S-A2p", "привязка b ко времени дня"), ("S-A2", "S-A0", "a и b вместе"))
THR = {"У1-S": "0,415", "У2-S": "0,50", "У3-S": "1,00"}


def word(d: float) -> str:
    return "лучше" if d < 0 else "хуже" if d > 0 else "равно"


def _split(tot: float, tie: float, flex: float) -> str:
    if not tot > 0:
        return "Разложение выигрыша A2 над A1 не определено: QLIKE(A2) не ниже QLIKE(A1) (разность %s)." % c_(tot, 6, True)
    return ("Разложение выигрыша A2 над A1 в единицах QLIKE: всего %s; на привязку ко времени дня (A2 против A2p) приходится %s (%s %%); остальное — %s (%s %%) — достаётся и контролю A2p с переставленным временем той же сложности (28 параметров)." %
            (c_(tot, 6), c_(tie, 6), c_(100 * tie / tot, 1), c_(flex, 6), c_(100 * flex / tot, 1)))


def _ab(T, h, b, lab):
    return stat(T, "%s|%s %s" % (h, b, lab), "ablation")


def sec4(T: dict) -> str:
    rows = []
    for h, b, what in ABL:
        for lab in ("F", "слой X1_A"):
            s = _ab(T, h, b, lab)
            rows.append(["%s − %s: %s" % (h, b, what), NAME[lab], c_(s["qlike_h"], 6), c_(s["qlike_b"], 6), c_(s["delta_pct"], 4, True), "[%s; %s]" % (c_(s["lo90"], 3, True), c_(s["hi90"], 3, True)), c_(s["p_dm"], 4), word(s["delta_pct"])])
    q = T["table"].set_index("series")["qlike"]
    tot, tie, flex = q["S-A1"] - q["S-A2"], q["S-A2p"] - q["S-A2"], q["S-A1"] - q["S-A2p"]
    s0, w0 = stat(T, "S-A0|X1"), stat(T, "S-A0|W5-G")
    return ("## 4. Абляции A0 → A1 → A2 → A2p\n\n" + dm_line(T) + " Δ — % QLIKE второго ряда (базы пары); p DM — NW-8, односторонний «первый лучше», БЕЗ поправки (абляции вне семейства из 12 гипотез).\n\n" +
            md(["Пара", "уровень", "QLIKE первого", "QLIKE второго", "Δ, %", "90 % интервал", "p DM", "первый"], rows) +
            "\n\nЧто дала память a: A1 − A0 = %s %% (%s). Что дали внутридневные веса b: A2 − A1 = %s %% (%s). A0 против X1 (переоценка корректора и HAR-части под QLIKE без новой структуры): слой Δ = %s %%, F против W5-G Δ = %s %%.\n\n" %
            (c_(_ab(T, "S-A1", "S-A0", "F")["delta_pct"], 4, True), word(_ab(T, "S-A1", "S-A0", "F")["delta_pct"]), c_(_ab(T, "S-A2", "S-A1", "F")["delta_pct"], 4, True),
             word(_ab(T, "S-A2", "S-A1", "F")["delta_pct"]), c_(s0["delta_pct"], 4, True), c_(w0["delta_pct"], 4, True)) +
            _split(tot, tie, flex))


def _guard(T, cand) -> str:
    g = T["guard"]
    g = g[(g["candidate"] == cand) & (g["id"] != "кандидат")]
    return md(["Условие", "пара", "значение", "порог", "выполнено"], [[r.id, r.pair, val(r.value), val(r.threshold), yn(r.passed)] for r in g.itertuples()])


def sec5(T: dict) -> str:
    g, sp = T["guard"], T["stats"]
    kand = {c: g[(g["candidate"] == c) & (g["id"] == "кандидат")]["value"].iloc[0] for c in ("S-A2", "S-A1")}
    m = sp[sp["kind"] == "main"]
    s2, s1 = stat(T, "S-A2|X1"), stat(T, "S-A2|W5-G")
    return ("## 5. Защита У1-S…У5-S и решение\n\n" + dm_line(T) + " Пороги — константы спецификации (У1-S %s %%, У2-S %s %%, У3-S %s %%); применяются к обеим парам: М1 (X1_A против X1) и М2 (F_A против W5-G); h1 — 2018–2021, h2 — 2022–2026. Порядок A2 → A1; A0 и A2p кандидатами не являются.\n\n" % (THR["У1-S"], THR["У2-S"], THR["У3-S"]) +
            "### S-A2\n\n" + _guard(T, "S-A2") + "\n\n### S-A1\n\n" + _guard(T, "S-A1") + "\n\nРешение (выведено кодом из строк выше): **" + g[g["id"] == "решение"]["value"].iloc[0] + "**.\n\n"
            "Раздельно:\n1. Техническая корректность — гейты листов IMPL и E.G1–E.G3 в `GATES_S.md` (отступления и ABANDON — там).\n"
            "2. Численное улучшение: S-A2 М1 Δ = %s %%, М2 Δ = %s %%; S-A1 М1 Δ = %s %%, М2 Δ = %s %%.\n" % (c_(s2["delta_pct"], 4, True), c_(s1["delta_pct"], 4, True), c_(stat(T, "S-A1|X1")["delta_pct"], 4, True), c_(stat(T, "S-A1|W5-G")["delta_pct"], 4, True)) +
            "3. Прохождение зарегистрированных ограничений: S-A2 — %s, S-A1 — %s (число выполненных из 10).\n" % (kand["S-A2"], kand["S-A1"]) +
            "4. Статистическая подтверждённость: подтверждённых гипотез из 12 — %d (условие: Δ < 0, p Холма N = 831 < 0,05 и p SPA < 0,05); наименьший p Холма (N = 831) = %s, наименьший p SPA (c) = %s.\n" %
            (int(m["verdict"].map(lambda v: "подтверждено" in v and "не подтверждено" not in v).sum()), c_(m["p_holm_N"].min(), 4), c_(m["p_spa_c"].min(), 4)) +
            "5. Независимое подтверждение: нет (только данные после 10.11.2026 на замороженном коде, процедура Л5–Л6).")


def _key(T, h, b, item) -> str:
    k = T["conc"]
    return str(k[(k["series"] == h) & (k["base"] == b) & (k["item"] == item)]["key"].iloc[0])


def sec6(T: dict) -> str:
    rows = []
    for c in ("S-A0", "S-A1", "S-A2"):
        for pid, h, b in (("М1", "X1_A(%s)" % c, "X1"), ("М2", c, "W5-G")):
            y1, y2 = _key(T, h, b, "no_top_day"), _key(T, h, b, "var_share_top_day")
            rows.append([c, pid, c_(cnc(T, h, b, "summary", "delta_total_pct"), 4, True), c_(cnc(T, h, b, "no_day", "2022-02-24"), 4, True), c_(cnc(T, h, b, "no_year", "2022"), 4, True),
                         c_(cnc(T, h, b, "no_top_day", y1), 4, True) + " (" + y1 + ")", c_(cnc(T, h, b, "var_share_top_day", y2), 3) + " (" + y2 + ")", c_(cnc(T, h, b, "summary", "share_best_year"), 3), c_(cnc(T, h, b, "summary", "share_two_best_years"), 3)])
    return ("## 6. Концентрация и чувствительность (только диагностика, не новая выборка)\n\n" + dm_line(T) + " Основная оценка — все 2198 дат. Столбцы: Δ полного периода; Δ без t⁺ = 2022-02-24; без всего 2022; без самого влиятельного дня (max |l_кандидат − l_база|); "
            "доля дисперсии разности от крупнейшего дня (d_(1) − d̄)²/Σ(d_t − d̄)²; доля выигрыша от лучшего периода и от двух лучших (периоды 2018…2025, 2026 до 27.02, 2026 F; доли определены при суммарном выигрыше лет > 0).\n\n" +
            md(["Кандидат", "пара", "Δ, %", "без 24.02.2022", "без 2022", "без самого влиятельного дня", "доля дисперсии крупнейшего дня", "доля выигрыша: лучший период", "два лучших"], rows) +
            "\n\nСамый влиятельный календарный год и вклады каждого периода — в `S_A_CONC.csv` (item `no_top_year`, `loyo`) и `S_A_YEARS.csv`.")
