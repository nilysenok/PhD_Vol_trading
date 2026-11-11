"""S-A-EVAL, документ (Ж-254 п. 16; `S_A_SPEC.md`, разд. 8): разделы 7–8 отчёта — параметры по годам, число обученных вариантов и подгонок, стоимость и воспроизводимость, ответ «что улучшило прогноз».
Публичное API: `sec7(T, ck)` — `ck` = ключ `S.A.eval` из CHECKPOINT (sha256 входов и кода, written_at); `sec8(T)` — ответ из таблиц: M10-представление (b), структура памяти (a), привязка ко времени, только сложность; качество измерения и разложение — не испытывались.
Текст строится только из таблиц `S_A_*.csv` и ключа CHECKPOINT."""
from __future__ import annotations

import numpy as np

from models_v2.s18.a_doc_fmt import c_, md, stat
from models_v2.s18.a_doc_s2 import word

COST = ("Стоимость (оценка кодера листа IMPL-3, не измерение CPU; `GATES_S.md`, I3.G7): A0 — порядка 4 с; A1, A2, A2p — порядка 5 мин каждый при 4 параллельных процессах. Спецификация (разд. 9) оценивала верхнюю границу ≤ 4 CPU-ч. "
        "Оценка этого этапа (SPA, StepM, бутстрап B = 10 000, MCS) — секунды; CPU-часы не замерялись.")


def _theta(P) -> list:
    return list(P.columns[P.columns.get_loc("d0"):P.columns.get_loc("dk") + 1])


def _a2_table(P) -> str:
    y = P[(P["variant"] == "A2") & (P["level"] == "year")]
    rows = []
    for r in y.itertuples():
        cn = float(np.sqrt(sum(getattr(r, "c_%s%d" % (k, i)) ** 2 for k in "sml" for i in (1, 2, 3, 4))))
        rows.append([r.key, int(r.n), c_(r.lam, 5), c_(r.om_s, 3), c_(r.om_m, 3), c_(r.om_l, 3), c_(cn, 3), c_(r.dw, 4, True), c_(r.dk, 4, True), c_(r.d_s, 4, True), c_(r.d_m, 4, True), c_(r.d_l, 4, True)])
    return md(["Год D", "переоценок", "λ (среднее)", "ω_s", "ω_m", "ω_l", "‖c‖", "δw", "δκ", "δ_s", "δ_m", "δ_l"], rows)


def _b_table(P) -> str:
    g = P[(P["variant"] == "A2") & (P["level"] == "b_grid")]
    ks = [k for k in g["key"] if k.endswith("память s") or k.endswith("память l")]
    sel = [k for k in ks if k.startswith("год 2018") or k.startswith("год 2025")]
    rows = [[r.key] + [c_(getattr(r, "b_t%d" % t), 3) for t in (0, 25, 50, 75, 100)] for r in g.itertuples() if r.key in sel]
    return md(["Год и память", "b(τ=0)", "b(0,25)", "b(0,5)", "b(0,75)", "b(1)"], rows)


def _nonzero(P) -> str:
    th, rows = _theta(P), []
    for v in ("A0", "A1", "A2", "A2p"):
        r = P[(P["variant"] == v) & (P["level"] == "refit") & P["lam"].notna()]
        nz = (r[th].abs() > 1e-8).sum(axis=1)
        rows.append([v, len(r), c_(nz.mean(), 1), int(nz.max())])
    return md(["Вариант", "переоценок с подгонкой", "ненулевых параметров из 28 (модуль > 1e-8), среднее", "максимум"], rows)


def sec7(T: dict, ck: dict) -> str:
    P, f = T["params"], T["params"][T["params"]["level"] == "fits"]
    fits = md(["Вариант", "подгонки", "число подгонок L-BFGS-B"], [[r.variant, r.key, int(r.n)] for r in f.itertuples()])
    sha = ", ".join("%s %s" % (k.replace("data/", "").replace("docs/otchety/", ""), v[:16]) for k, v in sorted(ck["inputs"].items()))
    code = ", ".join("%s %s" % (k, v[:16]) for k, v in sorted(ck["code"].items()))
    return ("## 7. Параметры, число обученных вариантов, стоимость, воспроизводимость\n\n" + "Параметры S-A2 по годам (среднее по месячным переоценкам года; θ — среднее по 5 seeds); все месяцы и варианты — `S_A_PARAMS.csv`. D — первый торговый день месяца (переоценка).\n\n" + _a2_table(P) + "\n\nb_k(τ) на сетке (показ: exp(h(τ)) по среднему c года, нормировано по 53 слотам; k = s — короткая, l — длинная память), S-A2:\n\n" + _b_table(P) +
            "\n\nЧисло реально обученных вариантов: 4 (S-A0, S-A1, S-A2, S-A2p) из 7 зарегистрированных; S-A3, S-A2s, S-A2g не запускались и в N не входят. Подгонки:\n\n" + fits + "\n\nЭффективное число параметров:\n\n" + _nonzero(P) +
            "\n\n" + COST + "\n\nВоспроизводимость (sha256, первые 16 знаков; ключ `S.A.eval` в `docs/otchety/harj_x1/CHECKPOINT.json`, written_at " + ck["written_at"] + "): входы — " + sha + "; код оценки — " + code + ".")


def sec8(T: dict) -> str:
    q = T["table"].set_index("series")["qlike"]

    def ab(h, b):
        s = stat(T, "%s|%s F" % (h, b), "ablation")
        return "%s %% (p DM %s, %s)" % (c_(s["delta_pct"], 4, True), c_(s["p_dm"], 4), word(s["delta_pct"]))
    m1, m2 = stat(T, "S-A2|X1"), stat(T, "S-A2|W5-G")
    return ("## 8. Ответ: что улучшило прогноз\n\n" + "Из таблиц разделов 1–7; все сравнения — исследовательские, на раскрытой истории (не OOS); p абляций — без поправки.\n\n"
            "- **Новое представление M10 (внутридневные веса b):** A2 − A1 = " + ab("S-A2", "S-A1") + "; но контроль с переставленным временем дня A2p даёт A2p − A1 = " + c_(100 * (q["S-A2p"] / q["S-A1"] - 1), 4, True) + " % (по QLIKE таблицы 1, без теста)" +
            ", а привязка ко времени (A2 − A2p) = " + ab("S-A2", "S-A2p") + ".\n"
            "- **Структура памяти (a):** A1 − A0 = " + ab("S-A1", "S-A0") + ".\n"
            "- **Качество измерения** — в пилоте не испытывалось; **разложение риска** — не испытывалось.\n"
            "- **Дополнительная сложность:** A0 (7 параметров) — QLIKE %s; A1 (16) — %s; A2 (28) — %s; A2p (28, время переставлено) — %s; число ненулевых параметров — раздел 7.\n" % (c_(q["S-A0"], 6), c_(q["S-A1"], 6), c_(q["S-A2"], 6), c_(q["S-A2p"], 6)) +
            "- **Против контрольных версий:** S-A2 к X1 (слой, М1) %s %% (p DM %s), F к W5-G (М2) %s %% (p DM %s); решение этапа — раздел 5.\n\n" % (c_(m1["delta_pct"], 4, True), c_(m1["p_dm"], 4), c_(m2["delta_pct"], 4, True), c_(m2["p_dm"], 4)) +
            "Вывод ограничен таблицами: численное улучшение, прохождение ограничений и статистическая подтверждённость различаются (раздел 5); независимого подтверждения нет.")
