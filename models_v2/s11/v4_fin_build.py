"""V4-FIN: сборка `HARJ_HYBRID_V4_FINAL.md` и `REPRODUCTION_V4.md` из таблиц `FIN_*.csv` (на диске) и статических разделов. Дат в документе нет (воспроизводимость: сборка дважды даёт тот же текст)."""
from __future__ import annotations

import json

from models_v2.s11 import v4_fin_doc as DOC, v4_fin_text as TX, v4_fin_text2 as TX2, v4_io as IO

STAGES = ["audit", "A", "B", "Ca", "Cb", "E", "DX", "CS", "NTP", "CLD", "RQC", "H", "I", "FIN"]
STATUS = ("**Статус:** исследовательский, DEVELOPMENT / HISTORICAL WALK-FORWARD 2017–02.2026, не независимый OOS; первая цифра — потолок; ACCEPTED не присваивается. Выбор — только `crit_panel` (M16 × цели 2017–2021, `hp_stats_sel.crit_panel`); "
          "окна 2022+ в выбор не входили. Все ряды — на общих датах (38601 строка панели 17 бумаг). Δ — разность среднего QLIKE по M16, в процентах уровня QLIKE K1 того же окна; минус = лучше базы. "
          "Примечание о номере: блок «адресная доработка» в коде и CSV назван Ж-224 — это Ж-225 (журнал, строка 08.11 04:50); перенумерация в коде не делалась (строки входят в хэши кэшей).")
REGIMES = """- **C8 (мягкие режимы)** — лучшая конструкция этапа C по crit_panel; средний вес на M дрейфует от 0,398 (2018) до 0,635 (2022): после 2021 ML получает больший вес (таблица выше; C_REPORT). Стоп этапа C — по условию (1).
- **Доверие гибриду π по состояниям (NTP).** На шоке 06.04.2018 доверие к гибриду растёт (NTP2 0,599, NTP3 0,652; в среднем за 2018 — 0,527 и 0,616), на шоке 24.02.2022 падает (NTP2 0,121, NTP3 0,149): HAR-J теряет роль после структурного излома 2022 (NTP_REPORT). Но в обеих группах состояний (π > 0,5 и π ≤ 0,5) вклад HAR-J по DM значимо не отличен от нуля ни в одном окне (NTP_STATES.csv).
- **После изломов (DX_FEATURES.csv).** В 28 днях после 06.04.2018 D̄ = −0,2844 (n = 336) против +0,0025 вне окна — HAR-J там сильно лучше ML; после 09.03.2020: +0,0144 против +0,0010 (HAR-J не лучше). Именно апрель 2018 даёт основной проигрыш мета-слоя к K1 в 2018 (DX_2018.csv: +0,0117 в месяце 2018-04 для C8; бумаги MTLR, OGKB, ALRS).
- **Калибровка после шоков (E, RQC).** Множитель b перескакивает: на 24.02.2022–31.03.2022 средний b у E1–E4, E6 равен 0,47–0,50 (обрезка), у E5 +0,10; у RQC3 ln a ≈ −0,18…−0,26 всегда (смещение Йенсена в объявленной формуле — отступление в RQC_REPORT); правило «устойчива ли калибровка после скачков» — нет.
- **Режимы как признак (C8, CUSUM H6).** Нестабильность режима (H6) ухудшает K8 (Δ crit_panel +0,00506; DM p = 0,010 против K8 на всей истории, H_REPORT)."""


def _sec(n: int, title: str, body: str) -> str:
    return "## %d. %s\n\n%s\n" % (n, title, body.rstrip("\n"))


def build(t: dict) -> str:
    cp = t["cp"]
    s = [DOC.first_line(t), "", "# HARJ_HYBRID_V4_FINAL — итог программы HAR-J Hybrid V4 (Ж-221, Ж-222, Ж-223, Ж-225), h = 1, 17 бумаг", "", STATUS, ""]
    s.append(_sec(1, "Итог в трёх частях (Ж-225 п. 9)", TX2.parts(t)))
    s.append(_sec(2, "Три лучшие архитектуры", "Правило отбора (объявлено до счёта, CHECKPOINT.json `FIN_rule`): три конфигурации из 46 с наименьшим crit_panel; K8 входит в 46. «go» — три условия Ж-225 п. 10: (1) crit_panel ≤ %.5f; (2) crit_panel без целей 2017 ≤ min(K2, K3, D2) − 0,0005; (3) то же без 1 %% дат с наибольшей |разностью QLIKE| к лучшему из K2/K3.\n\n%s\nСсылочные ряды (те же столбцы):\n\n%s\nПо окнам 2017, 2018, 2017–2021, 2022–2025, 2026 (источник — `FIN_ALL.csv`):\n\n%s\nКонфигурации, лучшие и K2, и K3 одновременно на окне выбора и на 2022–2025 (малые разности; после Холма не значимо ни одно: в `C_TESTS.csv` и `NTP_TESTS.csv` p_holm к K2 и K3 в основном семействе не ниже 0,145 при N = 694; ни одна не прошла «go»):\n\n%s\nГлавный критерий (Ж-225 п. 10) —устойчивое преимущество над самостоятельными K2, K3 и D2, а не только над HAR-J; результат за счёт 2017 или крайних дней успехом не считается (условия (2), (3)). Количественное объяснение — раздел 13." % (cp["go_threshold_B"], DOC.top3_block(t), DOC.ref_block(t), DOC.windows_block(t), DOC.both_block(t))))
    s.append(_sec(3, "Критерий выбора и находка о весах годов", TX2.year_weights(t)))
    s.append(_sec(4, "Все 46 архитектур (по возрастанию crit_panel; обе меры окна выбора)", "Столбцы: crit_panel (окно all), условия «go» (1)–(3), простое среднее QLIKE по M16 2017–2021, Δ QLIKE к K1, K2, K3, D2 (% уровня K1) на всей истории и на 2022–2025. Источник — `FIN_ALL.csv`; совпадение с таблицами этапов (`*_COMPARISON.csv`) проверяется в `--check`.\n\n" + DOC.all_block(t)))
    s.append(_sec(5, "Математические определения операторов", TX.DEFS))
    s.append(_sec(6, "Аудит данных", TX.AUDIT))
    s.append(_sec(7, "Oracle bound (диагностика, не критерий; M = √(K2·K3))", "Доли — части QLIKE лучшей одиночной модели; O1 — лучший постоянный вес задним числом, O2 — дневной оракул внутри [min(H,M); max(H,M)], O3 — O2 с множителем смещения, O4 — оболочка {H, K2, K3}. Источник — `A_ORACLE.csv`.\n\n" + TX2.oracle_block()))
    s.append(_sec(8, "QLIKE по 17 бумагам (SBER — диагностика, в выбор и в M16 не входит)", "Вся история (источник — `FIN_TICKERS.csv`):\n\n%s\n2022–2025:\n\n%s" % (DOC.tickers_block(t, "all"), DOC.tickers_block(t, "2022_2025"))))
    s.append(_sec(9, "SBER и ориентир 0,2081", "Ориентир 0,2081 — `hybrid_final_v1` (HAR + XGBoost A1_roll_126), воспроизведён из сохранённого ряда на тех же 2271 датах и тех же RV (0,20814; расхождение 4,0e-5, SBER_BENCH.csv); **это потолок**: c и λ выбраны по crit_wf на всей истории. Сравнение со всеми рядами на 2271 дате (источник — `FIN_SBER.csv`; Δ к ориентиру: минус — лучше ориентира):\n\n" + DOC.sber_block(t)))
    s.append(_sec(10, "Сравнение со всеми базовыми моделями", "K1 (HAR-J), K2 (XGBoost), K3 (LightGBM), D2 (LGBM-D2) и K8 (равновесная смесь) — в таблицах разделов 2 и 4 (каждая из 46 конфигураций против каждой базы, обе окна); DM и бутстрап — раздел 14."))
    s.append(_sec(11, "Динамика весов", "Средний вес на M (C9 — K2 + K3) по годам цели, источник — `C_REPORT.md` («Средний вес на M по годам»):\n\n%s\nB2: вес M растёт с 0,385 (2018) до 0,774–0,780 (2023–2025) — дрейф относительной точности (B_WEIGHTS.csv). CLD1: вес почти постоянен, 0,501 ± 0,021 в 2017–2021 (CLD_REPORT). K8 — фиксированные 0,5." % TX2.WEIGHTS))
    s.append(_sec(12, "Эффект режимов", REGIMES))
    s.append(_sec(13, "Причины выигрыша и проигрыша; почему гибрид не превосходит бустинги устойчиво", TX2.explain(t) + "\n\nHAR-J против √(K2·K3) по годам цели (M16; источник — `DX_YEARS.csv`; D̄ < 0 — HAR-J лучше):\n\n" + TX2.dx_block()))
    s.append(_sec(14, "Статистические проверки", "DM (НУ лаг 5, по панельному QLIKE по датам) и блоковый бутстрап дат; трёх лучших и K8 против K1, K2, K3, D2; Холм N = %d и N_wide = %d, семейство M16 × {2022–2025, вся}. Источник — `FIN_TESTS.csv`.\n\n%s\nMCS 90 %% (`FIN_MCS.csv`):\n\n%s" % (cp["N_declared_before_stats"], cp["N_wide"], DOC.tests_block(t), DOC.mcs_block(t))))
    s.append(_sec(15, "Реестр испытаний, ABANDON, N Холма", TX.REGISTRY + "\n\n" + TX.ABANDON + "\n\n" + TX2.registry_table(t)))
    s.append(_sec(16, "Результаты по годам (2017 и 2022–2025 отдельно)", "Средний QLIKE по M16 за год цели (источник — `FIN_YEARS.csv`; 2017–2021 — окно выбора, 2022–2026 — описательно; 2026 — 38 дат):\n\n" + DOC.years_block(t)))
    s.append(_sec(17, "CHECKPOINT и воспроизводимость", "`CHECKPOINT.json`: N_prior %d, N_declared_before_stats %d, N_wide %d, go_threshold_B %.6f, используется %s из %s конфигураций бюджета, FIN: конфигураций 0; ключи кэшей и хэши входов/кода — по этапам (`B_cache_key` … `I_cache_key`); FIN сверяет, что все parquet программы (sha256 %s…) и таблицы этапов (sha256 %s…) побитно прежние. Команды — `REPRODUCTION_V4.md`." % (cp["N_prior"], cp["N_declared_before_stats"], cp["N_wide"], cp["go_threshold_B"], cp.get("used_after_I"), cp["budget"], cp["FIN_prior_parquet_sha"][:12], cp["FIN_prior_tables_sha"][:12])))
    return "\n".join(s) + "\n"


def reproduction() -> str:
    cmds = "\n".join("python3 -W ignore -m models_v2.s11.run_v4 --stage %s\npython3 -W ignore -m models_v2.s11.run_v4 --stage %s --check" % (x, x) for x in STAGES if x != "DX") + "\npython3 -W ignore -m models_v2.s11.run_v4 --stage DX\npython3 -W ignore -m models_v2.s11.run_v4 --stage DX --check"
    return ("# REPRODUCTION_V4 — воспроизведение программы HAR-J Hybrid V4\n\nСтатус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок — порядок этапов (audit → A → B → Ca → Cb → E → DX → CS → NTP → CLD → RQC → H → I → FIN); "
            "стадия без `--check` ПИШЕТ выходы (кэши, таблицы, отчёты, реестр), `--check` только читает (стадии audit, A, B, Ca, Cb, E, CS, NTP, CLD, RQC — в песочнице `v4_sandbox`; DX, H, I, FIN — чтение с диска). Повторная запись идемпотентна (ключ кэша = хэш входов и кода, Д5).\n\n"
            "```\n" + cmds + "\n```\n\nТесты: `python3 -W ignore -m pytest tests/test_v2_v4_*.py -q -p no:cacheprovider`.\n\nВходы (только чтение): `data/regen/harj_panel_hybrid` (K1 `har_<T>.parquet`, K2/K3 `k23_<T>.parquet`, D2 `hyb_lgbm_p17.parquet`, `engine_panel_p17.parquet`, `daily_<T>.parquet`), M10-данные проекта. "
            "Выходы: `data/regen/harj_hybrid_v4/` (прогнозы 46 конфигураций, кэш), `docs/otchety/harj_hybrid_v4/` (таблицы этапов, `FIN_*.csv`, отчёты), `docs/otchety/HARJ_HYBRID_V4_FINAL.md`. Стадия FIN конфигураций не расходует, parquet не пишет; её `--check` пересобирает `FIN_*.csv` из прогнозов в памяти, сверяет с диском и с таблицами этапов (допуск 1e-9) и проверяет числа итогового документа.\n"
            "Ключи и хэши: `CHECKPOINT.json` (`FIN_prior_parquet_sha`, `FIN_prior_tables_sha`, `*_cache_key`).\n")


def facts() -> dict:
    return json.loads((IO.REP / "CHECKPOINT.json").read_text())
