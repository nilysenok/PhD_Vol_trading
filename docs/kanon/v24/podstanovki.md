# Подстановки канона v24 для проекта PhD_Vol_trading

Blok: Ж-025. Основа — код и список файлов ветки `kanon-v22-pereschet` (ebf488b, 27.10). Данные ≥ 2022 и `test_locked/` не открывались: маски составлены по именам файлов и по коду загрузчиков. Граница периодов из кода: разработка — по 2021-12-31 (`fastbt/sigma_e.py: LAST`), тест — 2022…2026 (`fastbt/s9.py: TEST_YEARS`, `TEST_CUT = 2026-02-03`).

## [ТЕСТОВЫЕ_ПУТИ]

Отдельной папки у тестовых данных почти нет: большая часть рядов лежит в общих parquet и CSV с датами 2017–2026. Поэтому правило трёхслойное, и честно сказано, что ловит каждый слой.

**Слой 1. Пути, закрытые целиком.** Хук PreToolUse (Read, Grep, Glob, Bash с `cat|head|tail|less|sed|awk|python -c|duckdb|sqlite3` и путём) отвечает deny и пишет попытку в `logs/audit.jsonl`.

| Маска | Что там | Откуда известно |
|---|---|---|
| `test_locked/**` | замки открытий, результаты главы 3 | `docs/otchety/handoff_main.md` |
| `docs/tables/strat_test/**` | 47 таблиц теста главы 4 | `git ls-files` |
| `docs/tables/**/*_test.csv`, `docs/tables/**/*_test_*.csv` | 33 файла в корне `docs/tables`, 5 в `dlya_teksta/` | `git ls-files` |
| `docs/tables/**/*20{22,23,24,25,26}*` | `tabl_4_final_*_2022_2026*.csv` и т. п. | `git ls-files` |
| `**/*_test.parquet`, `**/*_test.csv` | 21 файл в `data/regen/`, 2 в `data/regen_B_debug/`, `data/fastbt/s9/*_test.parquet` (наивная σ, RVI) | `git ls-files`, `fastbt/s9.py:149–152` |
| `data/regen/D25/bars1m/*_202[2-6].csv` | минутные бары тестовых лет | `fastbt/bars5m.py:46` (`{T}_{Y}.csv`) |
| `data/regen/R2/S/otkrytie/**` | логи открытия главы 3 | `handoff_main.md` |
| `docs/otchety/*test*`, `docs/otchety/*otkryt*`, `docs/otchety/*final_test*` | отчёты с тестовыми числами | имена в `git ls-files` |
| `moex_volatility/results/final/strategies/walkforward_v4/daily_positions_pre2026.parquet` | позиции 2022–2025 | `fastbt/data.py:22`, `fastbt/tests/test_core.py:46` |

**Слой 2. Общие файлы с датами ≥ 2022 — открывать только через фильтр даты.** Хук целиком их не закроет: разработке они нужны. Правило: агент не читает их напрямую (Read/`head`/`cat` — deny), код читает только через загрузчик реестра `zagruzit(ryad, do="2021-12-31")`. Образец — `fastbt/sigma_e.read_dev`: `filters=[("date","<=",LAST)]` и `assert max(date) ≤ LAST`. Линтер `scripts/kanon/lint_chteniya.py` (в Э3) ищет в изменённых файлах `read_parquet|read_csv` по этим путям без фильтра даты, нашёл — CI падает.

| Файл или маска | Ряд |
|---|---|
| `moex_volatility/results/final/strategies/data/ohlcv_daily_full.parquet` | дневные бары |
| `…/strategies/data/vpred_aligned.parquet`, `data/regen/sigma_pred_h1_final3_full.parquet` | σ-прогнозы на всём периоде |
| `data/regen/features_pit_v3.parquet` | RV и признаки |
| `moex_discovery/data/**/candles_10m/*.parquet` (`final_v3…v6`, `dataset_final`, `external`) | 10-минутки, индексы |
| `moex_discovery/data/external/macro/raw/key_rate_cbr.parquet`, `fastbt/rates/key_rate_cbr_vladelec.csv` | ставка |
| `data/mcftr_iss.csv`, `data/rgbitr_iss.csv` | индексы |
| `data/corporate_actions.csv`, `data/calendar/*.csv` | события и календарь (сами не результат, но тестовые даты в них есть) |

**Слой 3. Канарейка и аудит — то, что слоям 1–2 не видно** (скрипт сам читает тест, число попало в отчёт): в выводе агентов и в PR ищется контрольная строка; `scripts/kanon/krasnaya_komanda.py` сверяет. Канарейка кладётся в новый файл по маске слоя 1 (`data/kanareyka/kanareyka_test.csv`), тестовые данные не трогаются.

Исключения: открытие (`[ЗАМОРОЖЕННЫЕ_ФАЙЛЫ]`, скрипты открытия) запускает только владелец через `/otkrytie`.

❓ Корневой `README.md` (раздел результатов главы 4) и `docs/handoff.md` содержат числа теста — 4 строки с годами 2022–2025 в каждом. Закрыть их для агентов слоем 1? Умолчание — закрыть `README.md` для агентов семьи E (им он не нужен); `docs/handoff.md` оставить открытым, тестовые числа из него вынести в `docs/otchety/` (для pisar), срок 28.10.

## [ЗАМОРОЖЕННЫЕ_ФАЙЛЫ]

После тега `final-a-*` правка блокируется хуком (Edit/Write) и CI (`git diff <тег>..HEAD -- <список>` пуст).

| Файл | Почему |
|---|---|
| `docs/final/FINAL_A.md` (создаётся в Э6) | предрегистрация дизайна до открытия |
| `fastbt/designs/final/main.yaml`, `d_binary.yaml`, `d_cont.yaml` (v1) | дизайн главы 4, по которому было открытие |
| `models/designs/r2/r2_A_sborka.yaml` и `models/designs/r2/*` | дизайн сборки главы 3 |
| `docs/tables/strat_dev/*_final3.csv` (11 файлов) | эталоны разработки |
| `docs/tables/strat_dev_v3/arhiv_v3_0/**`, `docs/tables/strat_dev_v3/arhiv_3D_s_ekranom/**` (с `SHA256.txt`) | архивы эталонов |
| `docs/otchety/arhiv_do_audita/**` (с `SHA256.txt`) | состояние до аудита |
| `scripts/analysis/final_test_ch4_v2.py`, `scripts/analysis/final_test.py`, `scripts/analysis/final_test_r2.py`, `fastbt/final_test.py` | скрипты открытия |
| `docs/resheniya/resheniya.md` | журнал решений: только дописывание (не заморозка целиком) |

## [КОМАНДА_ТЕСТОВ]

Быстрый прогон затронутых модулей (Stop-хук и PostToolUse):

```
/usr/bin/python3 -m pytest -q -x --timeout=120 $(scripts/kanon/testy_dlya.sh <изменённые .py>)
```

`scripts/kanon/testy_dlya.sh` по имени модуля `fastbt/<m>.py` или `scripts/analysis/<m>.py` отдаёт `tests/test_<m>*.py` и `fastbt/tests/test_<m>*.py`; нет совпадений — `tests/kanon/` целиком; для `fastbt/engine*.py`, `fastbt/signals/*` — `fastbt/tests/`. `fastbt/tests/test_core.py` из быстрого прогона исключён до починки: он читает 2022–2023 (`test_smoke_gate`, сейчас `skip`) — инцидент «`test_core` читает тест». Python — системный 3.9.6, pytest — пользовательский (`~/Library/Python/3.9`). Плагина `pytest-timeout` может не быть — тогда без `--timeout` и с `timeout 300` снаружи.

## [КРИТИЧНЫЕ_МОДУЛИ]

| Тема | Модули и функции |
|---|---|
| Движок исполнения | `fastbt/engine.py`, `fastbt/engine_trend.py`, `fastbt/execlayer.py`, `fastbt/exec_e.py` (PR #290, ещё не в базе), `fastbt/margin.py` |
| Метрики | `fastbt/metriki_e.py`, `fastbt/s9.sharpe` (`s9.py:696`), `fastbt/stats.sharpe_window`, `fastbt/s9_stats.py` |
| Бутстрап (Ledoit–Wolf) | `fastbt/fast_boot.py` (`lw_starts`, `_lw_loop`, `lw_sharpe_diff`, `boot_sharpes`), `fastbt/s9_stats.lw_sharpe_diff`, `fastbt/final_test.lw_p` |
| Загрузчик рядов | `fastbt/data.py` (`daily_df`, корпсобытия), `fastbt/bars5m.py`, `fastbt/sigma_e.py` (`read_dev`, `KNOWN`), `fastbt/msp.py` (ставка) |
| Ключ кэша | `fastbt/s9.cache_key` (`s9.py:397`), `fastbt/masks._cached` (`masks.py:111`), `fastbt/bars5m._key`, `fastbt/cache.py`, `fastbt/dev.py` (хэш конфига и кода, стр. 218), `fastbt/fast_signals.py` (стр. 46, 61), `fastbt/fast_boot.py:151`, `fastbt/fast_sigma.py:57` |

## [ВХОДНЫЕ_РЯДЫ]

Реестр (`docs/kanon/v24/reestr_ryadov.yaml`, Э5) — по строке на ряд: имя, путь, столбец даты, отметка сдвига (последний известный день, как `sigma_e.KNOWN`), окно, SHA-256 файла.

| Ряд | Путь | Читает |
|---|---|---|
| Бары 1m | `data/regen/D25/bars1m/{T}_{Y}.csv` | `bars5m.py` |
| Бары 5m (производные, кэш) | `data/fastbt/bars5m/*.parquet` | `bars5m.py` |
| Бары 10m, акции | `moex_discovery/data/final_v6/candles_10m/stocks/*.parquet` (и `final_v3…v5`, `dataset_final/01_stocks`) | скрипты `moex_volatility/` |
| Бары 10m, внешние и индексы | `moex_discovery/data/final_v6/candles_10m/external/*`, `external/moex_iss/indices_10m/*` | `moex_volatility/` |
| Дневные бары | `…/strategies/data/ohlcv_daily_full.parquet` | `fastbt/data.py` |
| σ-прогнозы (разработка) | `data/regen/ch4_dev/sigma_{naive,harj_b_h1,harj_b_h5,hyb_h1,hyb_h5}_dev.parquet` | `sigma_e.py` |
| σ-прогнозы (полные) | `…/strategies/data/vpred_aligned.parquet`, `data/regen/sigma_pred_h1_final3_full.parquet` | `data.py`, `designs/final/main.yaml` |
| σ семьи E (производные) | `data/fastbt/sigma_e/*.parquet` | `sigma_e.py` |
| RV, признаки | `data/regen/features_pit_v3.parquet` | `sigma_e.py` |
| RVI | `data/fastbt/s9/rvi_1850.parquet` | `s9.py` |
| Ставка ЦБ | `moex_discovery/data/external/macro/raw/key_rate_cbr.parquet`; ряд владельца `fastbt/rates/key_rate_cbr_vladelec.csv` | `msp.py` |
| Индексы | `data/mcftr_iss.csv`, `data/rgbitr_iss.csv` | `bench.py` |
| Корпсобытия | `data/corporate_actions.csv` | `data.py` |
| Календарь | `data/calendar/{cbr,cpi_ru,cpi_us,dividends_actual,dividends_expected,expirations,fomc,imoex_rebal,imoex_rebal_official}.csv` | признаки |
| Позиции до 2026 | `…/strategies/walkforward_v4/daily_positions_pre2026.parquet` | `data.py` (тест; в реестр — с пометкой «тест») |

Константные отрезки (G22) проверяются на каждом ряду в пределах разработки; известное объяснение — ставка 2018–2021 до #267 (`bfill`).

## Воспроизведение (G23)

Шапка — не строка внутри CSV (это изменило бы таблицы), а сопроводительный файл `<таблица>.meta`: `v`, команда, SHA-256 входов, коммит. Три таблицы разработки:

| Таблица | Команда |
|---|---|
| `docs/tables/semya_e_proverka_sdvig.csv` | `/usr/bin/python3 -m fastbt.sigma_e` |
| `docs/tables/tabl_4_marzha_2018_2021.csv` | `/usr/bin/python3 -m fastbt.margin` (аргументы — из `.meta`) |
| `docs/tables/tabl_4_sharpe_vs_stavka_2018_2021.csv` | там же |
