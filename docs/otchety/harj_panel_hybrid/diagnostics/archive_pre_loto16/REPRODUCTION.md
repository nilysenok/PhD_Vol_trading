# REPRODUCTION — HAR-J Panel Hybrid, h = 1, обучение на 17 (Ж-217, Ж-220; лист P3c, G13)

Статус: исследовательский, DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; цифры — потолок. Все команды — из корня репозитория: `python3 -W ignore -m models_v2.s10.run_hp --stage <стадия> [--part ...] [--module ...] [--check]`. Без `--check` стадия считает и пишет, с `--check` — проверяет записанный выход (rc 0 = ok). Данные — `data/regen/harj_panel_hybrid/` (вне git); отчёты — `docs/otchety/harj_panel_hybrid/`. Входы — `data/iss_check/bars_10m_full/<T>.parquet` (28 бумаг), `index_10m/IMOEX.parquet`; хэши — `P0_HASHES.csv`. `models_v2/` в этой ветке git не отслеживается: SHA файлов кода — `FINAL_PANEL_CONFIG.json`.

## Порядок стадий (от сырых M10 до таблиц)
| № | Команда | Результат | Ожидаемый SHA-256[:16] |
|---|---|---|---|
| 1 | `--stage inv` | `P0_INVENTORY.csv`, `P0_CALENDAR.csv`, `P0_HASHES.csv` | — |
| 2 | `--stage daily` | `daily_<T>.parquet` (28 + IMOEX), SBER = Ж-214 побитно | — |
| 3 | `--stage har` | `har_<T>.parquet` (K1), SBER = `components.HAR` | — |
| 4 | `--stage shift`, `--stage port` | `P0_SHIFT.csv`, `P0_PORTABILITY.csv` | — |
| 5 | `--stage panel` | `panel_base.parquet` | 6e2988c6a8615ac6 |
| 6 | `--stage k4567` | `k4567.parquet` (K4–K7 на 28; K6, K7 берутся в диагностике 17), `engine_panel.parquet` | k4567 2ceb56c1e38df0b2; engine_panel 565a479b25033d1e |
| 7 | `--stage hyb --module xgb`, затем `--module lgbm` (счёт на 28) | кэши `hyb_c_parts.parquet` (β², Ĥ^m, Ĥ^u) и `hyb_a4_feats.parquet` (дизайн K2), которые P17 переиспользует (У-23); сами результаты на 28 в выводы не идут | hyb_xgb c864e31cfe764676; hyb_lgbm c5d7cbcb3a4c4948 |
| 8 | `--stage p17 --part panel` | `engine_panel_p17.parquet` (поперечные признаки по 17) | 64efbc21ae4ee182 |
| 9 | `--stage p17 --part k45 --module xgb`, `--module lgbm` | `k45_p17.parquet` (K1, K4, K5 при обучении на 17) | adcad498dc29fe00 |
| 10 | `--stage p17 --part hyb --module xgb`, `--module lgbm`; `--part summary` | `hyb_xgb_p17.parquet`, `hyb_lgbm_p17.parquet`, `P17_SUMMARY.csv` | xgb 4651022b40f8fb8f; lgbm 4ae8d821a0a93377 |
| 11 | `--stage p17 --check` | сверка | — |
| 12 | `--stage k23 --tickers <бумаги>` (17 основных; готовые не пересчитываются; ≈ 5–6 мин на бумагу) | `k23_<T>.parquet`; GCHE посчитан листом P1a до сужения по У-24 (текущий код отказывает для E11), воспроизведение — кодом P1a | AFLT 8d98c562e2cd4c4d, ALRS 7071de29c933b844, HYDR 973b7c980eb68fe2, IRAO 28b8f2c7cc5319fc, LKOH 5269a9be8a9a1553, LSRG c60d47e66016fd6a, MGNT d6f14e91b5d77f93, MOEX a89f0de6aa399459, MTLR c9eb0f915e3e2449, MTSS 566cc090f130405d, NVTK 5e53e45f0892c688, OGKB 15548893b1b42e9f, PHOR 8096255d60bba84f, RTKM 2652e66dc6b741d4, SBER d19679f827513aaf, TATN 7c4828f9affb65b1, VTBR ec9c927fbfd1a827, GCHE 8af0dc467cdacc53 |
| 13 | `--stage s17 --part sel` (первым, до любого p; объявляет N = 631 в `CHECKPOINT.json`) | `statistics_p17/SELECTION.csv`, `SELECTION.json` | — |
| 14 | `--stage s17 --part loto --module xgb`; `--part loto --module lgbm` (второй процесс — `--part loto-rev --module lgbm`), ≈ 46 мин | `loto_p17_xgb.parquet`, `loto_p17_lgbm.parquet` | loto_p17_xgb 0a75cbacec3f4dcb; loto_p17_lgbm cb0d121abd37ea0e (в `P3_17_REPORT.md` и G17 метки перепутаны — измерено 07.11.2026 по именам файлов) |
| 15 | `--stage s17 --part e11`, затем `--part stats` | `statistics_p17/{COMPARISON,TESTS,MCS,PER_TICKER,HARJ_VALUE,LEVELS,LOTO}.csv` | — |
| 16 | `--stage s17 --part all --check`, `--stage k23 --check` | сверка | — |
| 17 | `--stage diag` (правила R15 записаны в `CHECKPOINT.json` до счёта: `diag_rules_declared_at`, `diag_rules_sha`) | `diagnostics/CAUSES.csv` (+ `GAMMA_TICKER`, `LOTO_TICKER`, `YEAR_GAIN`, `CHOICE_DIST`), `diagnostics/FINAL_TABLES.md`, `FINAL_PANEL_CONFIG.json`, `docs/otchety/HARJ_PANEL_HYBRID_FINAL.md` (из `diagnostics/FINAL_TEXT.md` + таблицы) | CAUSES.csv см. ниже |
| 18 | `--stage diag --check` | сверка выводов, SHA правил, наличие документов | — |

SHA `CAUSES.csv` не фиксируется (шапка несёт время объявления правил, строки детерминированы) — при воспроизведении сверять строки и вывод «подтверждена / не подтверждена».

## Тесты
`python3 -W ignore -m pytest tests/test_v2_hp_*.py -q -p no:cacheprovider` (мутации: E11 в обучении/критерии/диагностике, выбор по 28, диагностика читает результаты на 28, Холм с N = числу строк, правила R15 изменены после объявления). Тесты не ходят в сеть.

## Что нельзя воспроизвести без оговорки
- Стадии 7, 10, 12, 14 занимают часы (K2/K3 ≈ 5–6 мин на бумагу; LOTO ≈ 46 мин в два процесса).
- GCHE `k23_GCHE.parquet` — только из архива данных (посчитан до У-24).
- Результаты на 28 (`statistics/`) — архив, в выводы не входят (Ж-220).
