"""V5 лист FIN, G24: сборка итогового документа из блоков (`v5_fin_doc*`, `v5_fin_build`, `v5_fin_text`) и запись `HARJ_HYBRID_V5_FINAL.md`, `REPRODUCTION_V5.md` (только стадия `FIN`, не `--check`)."""
from __future__ import annotations

from models_v2.s12 import v5_fin_build as B, v5_fin_doc as W, v5_fin_doc2 as W2, v5_fin_text as TX


def sec(n: int, title: str, body: str) -> str:
    return "## %d. %s\n\n%s\n" % (n, title, body.rstrip("\n"))


def build(t: dict) -> str:
    bad = B.claims_fail(t)
    if bad:
        raise ValueError("утверждения документа не выполнены на данных: %s" % bad)
    g1, s1, g = t["gs"], t["sp"], W.go_row(t)
    thr = g.iloc[0]
    parts = [W.first_line(t), "", TX.STATUS, "", "# HARJ_HYBRID_V5_FINAL — итог программы HAR-J Hybrid V5 (Ж-226), h = 1, 17 бумаг", ""]
    parts.append(sec(1, "Итог: условия «go» по всем рядам", "Пороги (CHECKPOINT.json): crit_panel ≤ %s; без целей 2017 ≤ %s; без 1 %% крайних дат ≤ %s (K8 в планке, запас 0,0005). Источник — `FIN_ALL.csv`; M — только справочный ряд.\n\n%s" % (f5(thr["thr1"]), f5(thr["thr2"]), f5(thr["thr3"]), W.crit_block(t))))
    parts.append(sec(2, "Лучшие конфигурации: Δ QLIKE к контролям (% уровня K1)", "%s (лучшая GS), `FIN_ALL.csv`:\n\n%s\n%s (лучшая SP):\n\n%s\nУровни QLIKE (M16) по окнам:\n\n%s" % (g1, W.delta_block(t, g1), s1, W.delta_block(t, s1), W.levels_block(t))))
    parts.append(sec(3, "Разбор критерия выбора (лист 0)", W2.criterion_block(t)))
    parts.append(sec(4, "Полные формулы", TX.FORMULAS))
    parts.append(sec(5, "Математическое отличие от V3/V4", TX.DIFFERENCE))
    parts.append(sec(6, "QLIKE по всем 17 бумагам (SBER — диагностика, в выбор не входит)", "Вся история (`FIN_TICKERS.csv`):\n\n%s\n2022–2025:\n\n%s\n2017–2021:\n\n%s\nЧисло бумаг с улучшением (QLIKE ниже) к K8 / к лучшему из K2, K3:\n\n%s" % (W.tickers_block(t, "all"), W.tickers_block(t, "2022_2025"), W.tickers_block(t, "2017_2021"), W.counts_block(t))))
    ref = W.f(float(t["sber"].loc[t["sber"]["window"] == "all", "ref_a1_roll_126"].iloc[0]), 4)
    parts.append(sec(7, "SBER и ориентир прежнего гибрида", "Ориентир " + ref + " — `hybrid_final_v1` на тех же датах, **потолок** (параметры выбраны на всей истории); не контроль новой панели. Δ к ориентиру: минус — лучше ориентира (`FIN_SBER.csv`).\n\n" + W.sber_block(t)))
    parts.append(sec(8, "Результаты по годам", "Средний QLIKE по M16 за год цели (`FIN_YEARS.csv`; 2017–2021 — окно выбора, 2022–2026 — описательно, 2026 неполный):\n\n" + W.years_block(t)))
    parts.append(sec(9, "Совместные ошибки", "e = ln(RV/F) на общих строках M16 (`FIN_JOINT.csv`).\n\n%s\n%s" % (W2.joint_pairs_block(t), W2.joint_all_block(t))))
    parts.append(sec(10, "Статистическая значимость", "DM (НУ лаг 5, панельный QLIKE по датам) и блочный бутстрап дат, лучшие GS и SP против K1, K2, K3, D2, K8; Холм N = N_declared и N_wide, основное семейство — окна 2022–2025 и вся (`FIN_TESTS.csv`).\n\n%s\nMCS 90 %% (`FIN_MCS.csv`):\n\n%s" % (W2.tests_block(t), W2.mcs_block(t))))
    parts.append(sec(11, "Стабильность: стресс-проверки и LOTO", "Стресс (`GS_STRESS.csv`, `SP_STRESS.csv`; ни один фильтр не подменяет главный результат):\n\n%s\nLOTO (`GS_LOTO.csv`, `SP_LOTO.csv`; среднее по бумагам, исключённая бумага не входит в обучение и межбумажные факторы):\n\n%s" % (W2.stress_block(t), W2.loto_block(t))))
    parts.append(sec(12, "Стоимость вычислений", "Замер `--check` стадий (`FIN_COST.csv`; прогнозы из кэша, не обучение с нуля):\n\n" + W2.cost_block(t)))
    parts.append(sec(13, "Абляция по цепочкам (a − b; минус = a лучше)", "Цепочки H → K8 → GS2 → GS3 / GS4 / GS5; K8 → SP1; SP2 → SP3; SP2 → SP4; H3 V4 (форма дня) против SP2, SP4 и K8 (`FIN_ABLATION.csv`; здесь H = K1):\n\n" + W2.ablation_block(t)))
    parts.append(sec(14, "Причины отклонения направлений", "Правила — план и `GATES_V5.md`; числа — из таблиц листов.\n\n" + B.closure_block(t)))
    parts.append(sec(15, "Особый вопрос: есть ли в новых архитектурах информация, которой нет в бустингах", B.special(t) + "\nОхват (`FIN_ENCOMPASS.csv`):\n\n" + W2.enc_block(t)))
    parts.append(sec(16, "Реестр всех испытаний и CHECKPOINT", W2.registry_block(t)))
    parts.append(sec(17, "Отступления листов", TX.DEPARTURES))
    parts.append(sec(18, "Воспроизводимость", TX.FOOT))
    return "\n".join(parts) + "\n"


def f5(x: float) -> str:
    return W.f(x, 5)


def reproduction() -> str:
    cmds = "\n".join("python3 -W ignore -m models_v2.s12.%s --stage %s\npython3 -W ignore -m models_v2.s12.%s --stage %s --check" % (m, st, m, st) for st, m in B.STAGES)
    return ("# REPRODUCTION_V5 — воспроизведение программы HAR-J Hybrid V5 (Ж-226)\n\nСтатус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок: 0 (аудит, критерий, паспорт) → DX1 (диагностики 1b) → GS (лист 1a) → SP (лист 2) → FIN (итог). "
            "Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы GS и SP пишутся в `data/regen/harj_hybrid_v5/` с ключом кэша = хэш входов и кода (Д5); FIN прогнозов не считает, конфигурации не расходует. "
            "GS: до GS3–GS5 нужно записать форму score (`--decide-form` после `--cfg GS1`, `--cfg GS2`); `--cfg GSn` считает одну конфигурацию.\n\n```\n%s\n```\n\nОтдельные шаги GS: `python3 -W ignore -m models_v2.s12.run_v5 --stage GS --cfg GS1` (и GS2), затем `--stage GS --decide-form`, затем `--cfg GS3`…`GS5`.\n\n"
            "Тесты: `python3 -W ignore -m pytest tests/test_v2_v5_*.py -q -p no:cacheprovider`.\n\nВходы (только чтение): прогнозы V4 (`data/regen/harj_hybrid_v4`, `harj_panel_hybrid`), M10-бары 17 бумаг. Выходы: `data/regen/harj_hybrid_v5/`, `docs/otchety/harj_hybrid_v5/` (таблицы, отчёты, `CHECKPOINT.json`, `ALL_TRIALS.csv`), `docs/otchety/HARJ_HYBRID_V5_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.\n" % cmds)


def write_docs() -> None:
    t = W.load()
    B.FINAL.write_text(build(t), encoding="utf-8")
    B.REPRO.write_text(reproduction(), encoding="utf-8")
