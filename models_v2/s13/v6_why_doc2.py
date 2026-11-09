"""V6 лист Z3: документ причин, разделы 4–7, первая строка («Вывод») и сборка `build(t)` — всё из таблиц (`v6_why_doc` — разделы 1–3, форматирование). Рекомендаций по V7 нет."""
from __future__ import annotations

from models_v2.s13 import v6_why_data as DT, v6_why_doc as W
from models_v2.s13.v6_why_doc import f, md, one, pr, s

NOT_CHECKED = """## Что не проверено
- Причинность: все разделы — измеренные свойства рядов; корреляции, наклоны и доли дней не доказывают, что именно они вызывают отсутствие устойчивого преимущества.
- Годовые Δ_y, вклады, терцили и скользящие окна даны без интервалов; интервал (90 %, блок 20, 2000, seed 0) есть только у разности Δ двух окон (раздел 6). Поправки на множественность для этих диагностик нет: они описательные, не отбор.
- Группы «по исходу» (раздел 3) используют будущее (RV(t⁺)) и не годятся как правило; пороги терцилей PIT на первых 250 датах не определены.
- Окна «2017–2021» и «2022–2025» заданы заранее; 2026 (38 дат) входит только в годовые строки и «вся история». Выбор окон после просмотра не менялся.
- Дизайн K2/K3 выбирался на SBER 2017–2021 (оговорка листа 0): сравнение с ними на этих годах не независимо.
- Не проверены другие тикеры, горизонты h = 5 и 22, другая цель RV, другие окна скользящей Δ и другие пороги крайних дней.
- Калибровка (раздел 4) — HAC-ошибки Ньюи–Уэста лаг 5 в матричной форме с теми же весами, что у DM (в `s3.dm` HAC есть только для среднего; при единственном регрессоре-константе формулы совпадают, тест).
- Доля на границе клипа для K2/K3 считается как доля дней с |ln(F/K1)| ≥ 1 − 1e-9; у бустингов клипа нет, это только сравнимая мера.
"""


def sec_mz(t: dict) -> list:
    out = ["## 4. Калибровка", "", "Mincer–Zarnowitz в логарифмах: ln RV(t⁺) = a + b·ln F, OLS, HAC-ошибки (Бартлетт, лаг 5) в скобках; t(b = 1) = (b − 1)/se_b. Сдвиг = среднее ln RV − ln F (минус = прогноз выше исхода в среднем).", ""]
    m = t["WHY_MZ"]
    z = m[m.kind == "mz"]
    out += md(["ряд", "окно", "дат", "a (se)", "b (se)", "t(b = 1)", "сдвиг"], [[r.series, r.window.replace("_", "–"), r.n, "%s (%s)" % (s(r.a, 3), f(r.se_a, 3)), "%s (%s)" % (f(r.b, 3), f(r.se_b, 3)), s(r.t_b_vs_1), s(r.shift, 3)] for r in z.itertuples()])
    ys = m[m.kind == "year"]
    ser = ["A2", "E2", "K3", "K2", "K1"]
    return out + ["Средний сдвиг ln RV − ln F по годам цели:", ""] + md(["ряд"] + DT.YEARS, [[x] + [s(one(ys, series=x, window=y)["shift"], 3) for y in DT.YEARS] for x in ser])


def sec_capture(t: dict) -> list:
    out = ["## 5. Сколько бустинговой поправки ловит гибрид", "", "x = ln(бустинг/K1), y = ln(F_гибрид/K1); наклон y на x (OLS), корреляция, доля дней с совпадающим знаком x и y (нулевой y не совпадает со знаком x ≠ 0)."]
    z = t["WHY_CAPTURE"]
    rows = [[r.base, r.hybrid, r.window.replace("_", "–"), r.n, f(r.slope, 3), f(r.corr, 3), f(r.same_sign_pct, 1)] for r in z.itertuples()]
    return out + [""] + md(["бустинг", "гибрид", "окно", "дат", "наклон", "корреляция", "совпадение знака, %"], rows)


def sec_roll(t: dict) -> list:
    out = ["## 6. Устойчивость во времени", "", "Скользящая Δ к базе по окну 250 смежных дат (шаг 1), в % уровня K1 того же окна: доля окон с Δ < 0, минимум, максимум. Разность средних Δ: 2022–2025 минус 2017–2021 (в % уровня K1 своего окна), 90 % интервал — блоковый бутстрап (блок 20, 2000, seed 0) по каждому окну отдельно; «0 внутри» — лежит ли 0 в интервале.", ""]
    r = t["WHY_ROLL"]
    rows = []
    for h, b in DT.PAIRS:
        a, c = one(r, hybrid=h, base=b, kind="roll"), one(r, hybrid=h, base=b, kind="split")
        rows.append([pr(h, b), int(a.n_windows), f(a.share_neg_pct, 1), s(a.min_pct), s(a.max_pct), s(c.delta_early_pct), s(c.delta_late_pct), s(c.diff_pct), "[%s; %s]" % (s(c.lo90), s(c.hi90)), "да" if c.lo90 <= 0 <= c.hi90 else "нет"])
    return out + md(["пара", "окон", "окон с Δ < 0, %", "мин. Δ, % K1", "макс. Δ, % K1", "Δ 2017–2021", "Δ 2022–2025", "разность", "90 % интервал", "0 внутри"], rows)


def sec_gsize(t: dict) -> list:
    out = ["## 7. Величина поправки", "", "|g| = |ln(F/K1)|: среднее, 95-й перцентиль и доля дней на границе клипа (|g| = 1, допуск 1e-9). У K2 и K3 клипа нет — доля дана как сравнимая мера.", ""]
    z = t["WHY_GSIZE"]
    sel = [(x, "all") for x in DT.HYB + DT.BOOST[::-1]] + [(x, y) for x in ("A2", "E2", "K3") for y in DT.YEARS]
    rows = [[x, "вся история" if w == "all" else w, int(one(z, series=x, window=w).n), f(one(z, series=x, window=w).mean_abs, 4), f(one(z, series=x, window=w).p95_abs, 4), f(one(z, series=x, window=w).at_clip_pct, 2)] for x, w in sel]
    return out + md(["ряд", "окно", "дат", "среднее |g|", "95-й перцентиль |g|", "на границе клипа, %"], rows)


def first_line(t: dict) -> str:
    parts = []
    for h, b in DT.MAIN:
        k = W.year_stats(t, h, b)
        tl = one(t["WHY_TAILS"], hybrid=h, base=b, frac=0.05)
        r, c = one(t["WHY_ROLL"], hybrid=h, base=b, kind="roll"), one(t["WHY_ROLL"], hybrid=h, base=b, kind="split")
        parts.append("%s: Δ вся история %s %% уровня K1, лет с Δ < 0 — %d из %d (2021 → 2022: %s → %s %%), без 5 %% крайних дней Δ = %s %%, окон 250 дат с Δ < 0 — %s %%, разность Δ 2022–2025 и 2017–2021 %s %% [%s; %s] (%s)" % (
            pr(h, b), s(k["total"]), k["neg"], k["n"], s(k["y21"]), s(k["y22"]), s(tl.delta_without_pct), f(r.share_neg_pct, 1), s(c.diff_pct), s(c.lo90), s(c.hi90), "0 внутри интервала" if c.lo90 <= 0 <= c.hi90 else "0 вне интервала"))
    return "Вывод (измеренные свойства, не причины): " + "; ".join(parts) + "."


def build(t: dict) -> str:
    head = [first_line(t), "", "**DEVELOPMENT / HISTORICAL WALK-FORWARD, диагностика, не отбор; первая цифра — потолок.** Ж-230, лист Z3; SBER, h = 1, 2271 общих дат; сохранённые прогнозы, обучения и новых конфигураций нет. Основные пары заданы заранее: A2 к K3 (по заданию — лучшая конструкция V6 за 2017–2021) и E2 к K3 (по заданию — лучшая по всей истории); остальные пары — в таблицах `docs/otchety/harj_hybrid_v6/why/WHY_*.csv`.", ""]
    body = W.sec_years(t) + W.sec_tails(t) + W.sec_regime(t) + sec_mz(t) + sec_capture(t) + sec_roll(t) + sec_gsize(t)
    return "\n".join(head + body + NOT_CHECKED.splitlines()) + "\n"
