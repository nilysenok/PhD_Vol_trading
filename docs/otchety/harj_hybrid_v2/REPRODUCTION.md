# REPRODUCTION — Ж-214 HAR-J Hybrid V2, SBER h=1

Статус: DEVELOPMENT; HISTORICAL WALK-FORWARD 2017-01-03…2026-02-27 (2271 дата цели, просмотрено, не независимый OOS); все числа — потолок. Рабочее дерево: ветка `gl3-r6-h5-dev` (код листов 1–5 в `models_v2/s9/` и `tests/test_v2_hv2_*.py`; коммит дирижёра `98daebe4` на ветке `harj-hybrid-v2` — листы 1–4).

## Порядок (из корня репозитория; стадии по одной, `run_in_background`; сеть не нужна)
1. `python3 -W ignore -m models_v2.s9.run_hv2 --stage ctrl` — контроли и A0 (HAR 0,2337 · XJ 0,2113 · LF 0,2117 · A1_roll_126 0,2081; A0 побитно).
2. `--stage m10` → `data/regen/harj_hybrid_v2/m10_daily.parquet`; `--stage diagv`; `--stage ledger` (реестр 541 строка побайтно).
3. Лист 2: `--stage q`, `--stage h`, `--stage h2`. Лист 3: `--stage dec`, `--stage rec`, `--stage sea`, `--stage dec2`, `--stage sea2`. Лист 4: `--stage ml`, `--stage mkt`. Ряды σ̂² — `data/regen/harj_hybrid_v2/*_series.parquet` (кэш по SHA кода и входов, Д5).
4. Лист 5: `--stage final` — объявляет N в `CHECKPOINT.json` (`N_declared_before_final` = 585), выбирает финалистов по `crit_wf`, считает тесты, MCS, уровни, лестницу, таблицу п. 29, рейтинг, `FINAL_V2_CONFIG.json`. Время счёта — минуты на одном процессе (точно не измерялось).
5. Тесты: `python3 -W ignore -m pytest tests/test_v2_hv2_*.py -q`.

## Входы и хэши (SHA-256)
Полный список хэшей входов и рядов — `FINAL_V2_CONFIG.json`, ключ `hashes` (components `0b18be29…ef2fc`, m10_daily `3373172e…7522`, ALL_TRIALS, 10 файлов `*_series.parquet`). Данные только из списка задания: `data/iss_check/{bars_10m_full,index_10m}`, `data/regen/xgb_lgbm_full_v1/`, `data/regen/hybrid_struct_v1/ERROR_STRUCTURE_DAILY.parquet` (режимы), `data/regen/harj_hybrid_v2/`.

## Параметры статистики
DM — Ньюи–Уэст, лаг 5, двусторонний; блочный бутстрап — блок 20, 5000, seed 42, ДИ 95 %; MCS — 90 %, блок 20, 5000, seed 42; Холм — `stats_d.holm_n(p, N)`, N = 541 + 44 = 585, семейство — тесты A и B всех 10 финалистов по всей истории (20 p-значений); период и срезы — 2017–21, 2019–21, 2022, 2023–25, 2026, вся история, без 24.02.2022, без 5 крупнейших дней.

## Выходы
`docs/otchety/harj_hybrid_v2/statistical_validation/` (FINALISTS, FORECAST_COMPARISONS, ANNUAL, REGIME, LARGEST_ERRORS, ERROR_CORRELATION, STATISTICAL_TESTS, LEVELS, TOP5_DAYS, MCS, ARCH_RANKING, FINAL_TABLE_P29, REFERENCE_ROBUST_EX_2402, REFERENCE_TESTS), `ABL_LADDER.csv`, `FINAL_V2_CONFIG.json`, `CHECKPOINT.json`, итог — `docs/otchety/HARJ_HYBRID_V2_FINAL_RESEARCH.md`.

## Известные ограничения воспроизведения
- Стадии листов 2–4 используют кэш рядов (ключ — SHA исходников модулей, M10-таблицы и components); смена кода пересчитывает ряд.
- Режимы (`REGIME.csv`) читают `ERROR_STRUCTURE_DAILY.parquet` Ж-212 (не пересчитывается здесь).
- Перенос на MOEX и панель в этом пакете не запускались.
