# P0_REPORT — лист P0 панельного гибрида (Ж-215 / Ж-217), 07.11.2026

Статус: DEVELOPMENT; HISTORICAL WALK-FORWARD, не независимый OOS; ML-обучений нет; первая цифра — потолок. Код — `models_v2/s10/`, тесты — `tests/test_v2_hp_*.py`, данные — `data/regen/harj_panel_hybrid/` (не в git).

**Проверка сдвига (Д1), первой строкой:** `P0_SHIFT.csv` — 518 рядов (29 объектов: 28 бумаг + IMOEX; 17–18 входных рядов каждого, в т. ч. K1 σ̂² на дате информации d), corr(x_d, ln RV_{d+k}), k = −2…+1. Максимум |corr| на k = 0 у 513; 5 несовпадений — RGSS corr, RGSS beta, SELG corr, UTAR corr, UTAR beta (|corr| ≤ 0,11, пик отличается от k = 0 меньше 2/√n) — слабые ряды, не утечка (`P0_SHIFT_MISMATCH.csv`). Ограничение: на слабых рядах проверка не различает сдвиг; защита от сдвига — тесты обрезки (G4/G5).

## Реестр гейтов: закрыто 6 из 6 (G1–G6); ABANDON нет
| Гейт | CHECK | Результат |
|---|---|---|
| G1 | `--stage inv --check` | rc 0; 28 бумаг, 2944 даты календаря; хэши входов совпали |
| G2 | `--stage daily --check` | rc 0; 29 файлов; тождества ≤ 4,3e-16; SBER = Ж-214 `m10_daily` (31 колонка) и `components.rv`: max_abs_diff = 0,0 |
| G3 | `--stage shift --check` | rc 0; см. строку выше |
| G4 | `--stage har --check` | rc 0; SBER = `components.HAR` на 2271 из 2271 строк, max_abs_diff = 0,0; обрезка будущего по боевому пути (бары после 30.06.2021 повреждены): 0,0 |
| G5 | `pytest tests/test_v2_hp_*.py -q` | rc 0; 23 passed (34 с); `--stage port --check` rc 0 |
| G6 | `--stage panel --check` | rc 0; `panel_base.parquet`: 59 146 строк × 85 колонок, SHA-256[:16] **6e2988c6a8615ac6** (полный 6e2988c6a8615ac6b45f5e5d99d369e0d9abbdf21f22f978faacd1a1d441741e); перезапуск — тот же SHA |

Панель: M16 36 346 строк, SBER 2 272, E11 20 528; пар `gap` (> 5 суток) 179, из них через остановку 2022 — 27 (по одной у бумаги, торговавшей в феврале 2022). `CHECKPOINT.json`: N_registry_start 585, budget 24 (10/10/4), used 0, leaf P0.

## Входы (SHA-256[:16])
SBER `bars_10m_full/SBER.parquet` b451cb48473ac15c; IMOEX cadc3e41cda11533; `components.parquet` 0b18be29aff34f8e; `m10_daily.parquet` Ж-214 3373172e162ffeb6; остальные 27 бумаг — `P0_HASHES.csv`. Эталон M10-величин SBER — `m10_daily.parquet` (ряды `*_series.parquet` Ж-214 — прогнозы моделей, не M10-величины; не использовались).

## Что сделано
- `hp_daily` — M10-величины, синхронные с IMOEX, β, идиосинкратическая часть тем же кодом `hv2_m10.build` / `hv2_sync.build` / `daily.build_daily` (без правки `s9/`); `hp_har` — K1 по R3 (код `har.py`), 28 бумаг + IMOEX; `hp_sync` — поперечные величины только по бумагам с днём t; `hp_feat` — X/M/C и z по прошлому бумаги; `hp_panel` — панель и `CHECKPOINT.json`; `hp_inv`, `hp_shift`, `hp_port` — инвентарь, Д1, переносимость; `hp_synth` — синтетика тестов; `run_hp` — стадии `inv|daily|har|shift|port|panel` с `--check`.
- Отчёты: `P0_INVENTORY.csv`, `P0_CALENDAR.csv`, `P0_SCHEDULE.csv`, `P0_HASHES.csv`, `P0_IDENTITIES.csv`, `P0_SBER_CHECK.csv`, `P0_K1.csv`, `P0_SHIFT.csv`, `P0_SHIFT_MISMATCH.csv`, `P0_PORTABILITY.csv`, `P0_FEATURES.md`.

## Мутации (мутант падает, боевой код проходит; через боевой путь бары → daily → K1 → признаки)
чужой день t+1 (`test_mutaciya_chuzhoj_den_t_plus_1_lovitsya`); ffill отсутствующей бумаги (`test_mutaciya_ffill_otsutstvuyushchej_lovitsya`); нормировка с днём t / с будущим (`test_mutaciya_normirovka_s_dnem_t_lovitsya`, `…_s_budushchim_lovitsya`); цель на календарный t+1 (`test_mutaciya_celi_na_kalendarnyj_t_plus_1_lovitsya`); HAR-J по полной выборке, класс № 104 (`test_mutaciya_har_po_polnoj_vyborke_lovitsya`); сдвиг ряда на +1 (`test_shift_plus_one_lovitsya_i_chestnyj_ryad_prohodit`); части дня не в сумме (`test_tozhdestva_i_mutatsiya_chasti_ne_v_summe`). Проверка по известному ответу: независимый `lstsq` (`test_k1_znanyj_otvet_nezavisimyj_ols`), z-оценка на чередующемся ряду.

## Отступления
1. **Сетка K1 шире A0 на пары через остановку 2022** (R2: `gap` входят в основной счёт): +1 строка на бумагу (SBER 2 272 вместо 2 271); на общих 2 271 строках — побитно.
2. **Д1 для прогноза K1**: хранится на дате информации d, ожидаемый максимум k = 0 (на дате цели был бы k = −1 — σ̂_d построен из RV_d; первый прогон дал 29 «несовпадений» именно по этой причине, выравнивание исправлено, не критерий). Критерий «шум» (пик − |corr_0| < 2/√n) — мой, для разбора несовпадений.
3. **Правило переносимости K2/K3** (≥ 126 дней до 01.01.2015 для годового LOYO и т. д.) — выбрано мной до счёта из входов SBER (~150 дней), не подбиралось под результат; PHOR (125) и TRMK (123) — на границе.
4. Формулы R5 уточнены там, где текст неполон (`P0_FEATURES.md`, раздел «Оговорки»); `--check` проверяет записанный выход (пересчитываются только тождества, сверка SBER и обрезка будущего).
5. Сектор — ручной справочник по общеизвестной принадлежности (не сверялся с внешним источником; MSNG, TGKA, MSRS, SELG — по памяти, ❓ проверить).

## Находки
- E11 тонкие: дней без «штатного» расписания до 1 499 (MSRS), дыр относительно IMOEX до 1 931 (UTAR; 864 дня, 568 строк панели), порог дня — лишь ≥ 30 баров из 53; RV по 30–40 барам менее сопоставима. QLIKE K1 (все даты, ср. по бумагам): E11 0,402, M16 0,296, SBER 0,2345, IMOEX 0,259 — **потолок** (K1 на тех же данных; для K2–K7 не сравнивалось).
- `c_sys_share` без верхней границы (хвост z до 450).
- K2/K3 переносимы (по входам) на 17 из 28; K6 — на 23; код K2/K3 жёстко на SBER (`prov.BARS`) — в P1 нужна обёртка. Не переносимы K2/K3: GCHE, LSRG, MRKC, MRKP, MSRS, PHOR, RGSS, SELG, TGKA, TRMK, UTAR (история; у MRKP, MSRS, RGSS, SELG, UTAR ещё < 252 пар к 2017); K6 — MRKP, MSRS, RGSS, SELG, UTAR.

## Требует решения (предложения У-n, коммитит дирижёр до P1)
- **У-1**: зафиксировать `P0_FEATURES.md` как уточнение R5 (J: ln(1 + J/RV_m); отклонения долей от 22 предыдущих дней; ln CLOSE/MIDDLE по частям RV; ε = 0,01 в c_ln_idio).
- **У-2**: уточнить R2/R3: сетка K1 включает пары через остановку 2022 (флаг `cross_seg`, `gap`); SBER на общих датах побитно = Ж-214.
- **У-3**: правило переносимости K2/K3 — оставить (17 бумаг) или снизить порог истории до ≥ 120 дней (+ PHOR, TRMK = 19) — решить до P1.
- **У-4**: для E11 — оставить порог ≥ 30 баров или ввести отметку «неполный день» (n_bars < 45) как признак/срез; сейчас без изменений.
- **У-5**: Д1 для прогнозных рядов — «последний известный день» = дата информации d (закрепить в R2).
- Не сделано: ничего из задания; K2–K7 и гибриды — листы P1–P2.

Проверено командами (все rc = 0): `run_hp --stage inv|daily|har|shift|port|panel --check`; `pytest tests/test_v2_hp_*.py -q -p no:cacheprovider` (23 passed); `python3 -m py_compile models_v2/s10/*.py tests/test_v2_hp_*.py`.
