"""V6 лист FIN, G52: сборка `HARJ_HYBRID_SBER_V6_FINAL.md` и `REPRODUCTION_V6.md` из таблиц на диске. Времена в документе — только моменты решения об A0g из журнала (подраздел 5.1, `v6_fin_z1`); сборка дважды даёт тот же текст. Утверждения текста, зависящие от данных, проверяет `claims_fail`: не выполнены — сборка отказывает."""
from __future__ import annotations

from models_v2.s13 import v6_0_io as IO, v6_fin_data as D, v6_fin_doc as W, v6_fin_doc2 as W2, v6_fin_doc3 as W3, v6_fin_text as TX, v6_fin_z1 as Z1

FINAL = IO.ROOT / "docs/otchety/HARJ_HYBRID_SBER_V6_FINAL.md"
REPRO = IO.REP / "REPRODUCTION_V6.md"
STAGES = ["0", "1a", "1b", "2", "3", "4", "5", "6", "FIN"]
SECTIONS = ["Результат", "Что исследовано в V1–V5", "Чем V6 отличается", "Формулы", "Полная таблица конфигураций", "QLIKE по годам", "Сравнение с контролями", "Абляции", "Стресс-проверки",
            "Разложение причин", "Реестр испытаний", "Решение о перспективах", "Отступления программы"]


def sec(n: int, title: str, body: str) -> str:
    return "## %d. %s\n\n%s\n" % (n, title, body.rstrip("\n"))


def claims_fail(t: dict) -> list:
    return W3.claims_fail(t) + Z1.claims_fail(t)


def ref_2081(t: dict) -> float:
    b = t["best"]
    return float(b[(b["section"] == "best") & (b["window"] == "all") & (b["series"] == "J210a:A1_roll_126")]["qlike"].iloc[0])


def map_tables(t: dict) -> tuple:
    one, two = t["map"].split("## 2.")
    pick = lambda x: "\n".join(ln for ln in x.splitlines() if ln.startswith("|")) + "\n"
    return pick(one), pick(two)


def build(t: dict) -> str:
    bad = claims_fail(t)
    if bad:
        raise ValueError("утверждения документа не выполнены на данных: %s" % bad)
    b17, ba = W.best(t, "2017_2021"), W.best(t, "all")
    m1, m2 = map_tables(t)
    ref, noise = ref_2081(t), t["cp"]["0"]["noise"]["mde_95_pct_of_k1"]
    inf = lambda src: W2.info_block(t, src)
    parts = [W.first_line(t), "", TX.STATUS, "", "# HARJ_HYBRID_SBER_V6_FINAL — итог программы HAR-J Hybrid V6 (Ж-227, Ж-229), SBER, h = 1", ""]
    parts.append(sec(1, "Результат: лучшие конструкции V6 против K1, K2, K3, K8, V4:NTP2", "Лучшая по QLIKE 2017–2021: **%s**; лучшая по всей истории: **%s** (Δ в %% уровня K1, 90 %% интервал, DM p; `FIN_TESTS.csv`). Статус уровня 5 — словами блока: %s.\n\n%s\n%s\nОхват (`FIN_ENCOMPASS.csv`):\n\n%s\nMCS 90 %% (`FIN_MCS.csv`):\n\n%s"
                    % (b17, ba, W.level5(t, b17), W.best_block(t, b17), "" if ba == b17 else "%s:\n\n%s\n" % (ba, W.best_block(t, ba)), W.enc_block(t, list(dict.fromkeys([b17, ba]))), W.mcs_block(t))))
    parts.append(sec(2, "Что исследовано в V1–V5 (`MAP_DONE_V6.md`) и лучший актуальный результат SBER", "%s\nЛучший актуальный PIT-сопоставимый ряд SBER — V4:NTP2: QLIKE вся история %s, 2017–2021 %s. Ориентир %s (Ж-210, A1_roll_126) — **потолок**: он выбран по `crit_SBER` на тех же целях 2017–2021 среди родственных правил."
                    % (m1, W.f(W.q(t, "V4:NTP2", "all")), W.f(W.q(t, "V4:NTP2", "2017_2021")), W.f(ref))))
    parts.append(sec(3, "Чем V6 отличается от прежних программ и что исключено как эквивалентное", "%s\nИсключено как эквивалентное: B — утро / вечер ≡ Ж-214 `rv_open`/`rv_close` и V4 C1–C4; C — поправка по √RQ ≡ HARQ-J (Ж-214 H1–H11); D — ядро только по времени ≡ V4 E1/E2; F — штраф нормы g ≡ Ж-212 Cb/Ce." % m2))
    parts.append(sec(4, "Формулы всех конструкций и уточнения, объявленные до счёта", TX.FORMULAS + "\n\n**Уточнения и добавления по листам:**\n\n" + TX.REFINEMENTS))
    parts.append(sec(5, "Полная таблица конфигураций (18 мест бюджета) и метрики по окнам", "Посчитано 18 из 18; условные A3, C4, E3 посчитаны по правилу; CB2 не считалась (её условие — новая информация E1 или E2 против A0 и D2 против K9 — не выполнено); A0g — диагностический контроль (Ж-230), отступление от предрегистрации. Бюджет: %s Подробнее — подраздел 5.1.\n\n%s\nУсловия «go» (`*_GO.csv`):\n\n%s\n%s\n%s"
                    % (Z1.budget_line(t), W2.configs_block(t), W2.go_block(t), "".join("QLIKE, RMSE, MAE, R², bias — %s (`FIN_ALL.csv`):\n\n%s\n" % (W.WN[w], W2.metrics_block(t, w)) for w in ("all", "2017_2021", "2022_2025", "2026")), Z1.section(t))))
    parts.append(sec(6, "QLIKE SBER по годам цели 2017…2026 (все конфигурации и контроли; D2 (контроль) — панельный V4/V5)", W2.years_block(t)))
    parts.append(sec(7, "Сравнение с HAR-J (K1), XGBoost (K2), LightGBM (K3), K8 и лучшим прежним гибридом (V4:NTP2)", "%s\nDM — односторонний «конфигурация лучше», Холм N = 722 и N_wide = 3122. Охват всех конфигураций:\n\n%s" % (
        "".join("%s\n%s\n" % (W2.delta_win(t, w), W2.p_win(t, w)) for w in ("2017_2021", "all", "2022_2025")), W.enc_block(t, D.CFG))))
    parts.append(sec(8, "Абляции", "**Правило «новая информация» по всем парам** (окно 2017–2021, `FIN_INFO_ALL.csv`):\n\n%s\n**Обе версии: против A0 и против A0g** (отступление, вопрос 1):\n\n%s\n**Признаки против обучателя** (`FIN_ABLATION.csv`):\n\n%s\n**DX-A** (задним числом, не выбор):\n\n%s"
                    % (inf(["A_INFO.csv", "C_INFO.csv", "D_INFO.csv", "E_INFO.csv", "F_INFO.csv", "G_INFO.csv"]), W2.versions_block(t), W2.decomp_block(t), W2.dxa_block(t))))
    parts.append(sec(9, "Стресс-проверки", "Δ к K1, %% (`FIN_STRESS.csv`; без 2017, без 2017–2018, без 1 %% крайних, кризисные и спокойные дни, верхние 10 %% H):\n\n%s\nΔ к K8, %%:\n\n%s\nСоседние точки (λ, число деревьев, k, κ; не конфигурации):\n\n%s\nДни высокой u (C, окно 2017–2021, против A0):\n\n%s"
                    % (W2.stress_block(t, "d_pct_K1"), W2.stress_block(t, "d_pct_K8"), W2.sens_block(t), W2.unst_block(t))))
    parts.append(sec(10, "Разложение причин успеха и неудач (только измеренные свойства)", W3.causes(t)))
    parts.append(sec(11, "Реестр испытаний, контрольные хэши, CHECKPOINT", W3.registry_block(t) + "\nУровни 1–5 по всем конфигурациям (`FIN_LEVELS.csv`):\n\n" + W.hdr(["конфигурация", "уровень", "выполнен", "основание"]) + "".join("| %s | %s | %s | %s |\n" % (r.config, r.level, r.passed, r.basis) for r in t["levels"].itertuples())
                    + "\nРазличимый эффект на SBER 2017–2021 (лист 0): %s %% уровня K1." % W.f(noise, 2)))
    parts.append(sec(12, "Решение о перспективах дальнейшей разработки", W3.perspectives(t)))
    parts.append(sec(13, "Отступления программы, ошибки дирижёра и оговорки", TX.DEPARTURES + "\n\n**Оговорки:**\n\n" + TX.CAVEATS.replace("{ref}", W.f(ref))))
    return Z1.label_tables("\n".join(parts) + "\n")


def reproduction() -> str:
    cmds = "\n".join("python3 -W ignore -m models_v2.s13.run_v6 --stage %s\npython3 -W ignore -m models_v2.s13.run_v6 --stage %s --check" % (s, s) for s in STAGES)
    return ("# REPRODUCTION_V6 — воспроизведение программы HAR-J Hybrid V6 (Ж-227), SBER, h = 1\n\nСтатус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок: 0 (аудит, пороги) → 1a (A0) → 1b (A1–A3, DX-A) → 2 (C1–C4) → 3 (E1–E3) → 4 (D1–D3, K9) → 5 (CB1, F1, F2) → 6 (A0g) → FIN (итог). "
            "Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы пишутся в `data/regen/harj_hybrid_v6/`; FIN прогнозов не считает и конфигурации не расходует.\n\n```\n%s\n```\n\n"
            "Тесты: `python3 -W ignore -m pytest tests/test_v2_v5_*.py tests/test_v2_v6_*.py -q -p no:cacheprovider`.\n\nВходы (только чтение): сохранённые прогнозы прежних программ SBER и M10-бары SBER (`data/iss_check/bars_10m_full/SBER.parquet`). Выходы: `data/regen/harj_hybrid_v6/`, `docs/otchety/harj_hybrid_v6/` (таблицы, отчёты листов, `CHECKPOINT.json`, `ALL_TRIALS.csv`), "
            "`docs/otchety/HARJ_HYBRID_SBER_V6_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.\n" % cmds)


def write_docs() -> None:
    FINAL.write_text(build(W.load()), encoding="utf-8")
    REPRO.write_text(reproduction(), encoding="utf-8")
