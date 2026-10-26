# Передача R2 — агент B (26.10, 17:01 по Маку)

Ветка `r2-A1-kesh-h1` (PR #185 → `kanon-v22-pereschet`, в main не сливать). Журнал — `docs/otchety/dev_log_r2_B1.md`, пп. 25–35 (протокол v3). Фоновых процессов нет: на 17:01 `ps` пуст, всё запускалось на переднем плане.

## Что сделано
- **`scripts/analysis/r2_roli.py`** — кластер 4 (роли none / feature / residual / offset / residual_std; resid_source in_sample / cross_fit; lib; grid current / resid_small_v1; early_stop native / qlike_total; monotone; objective; clip_sigma; shrink), кластер 12 (prob scale_boost, quantile5), кластер 15 (boost_refit yearly / monthly / quarterly_warm). Определения, оставленные схемой исполнителю, и таблица «роль × h → точки R1» — в `models/designs/r2/r2_B_roli.yaml` (❓ — умолчания).
- **Ядро S:** маски эмбарго `train_mask` / `val_mask` / `inner_masks`, `load_d0`, `shift_check`, `leak_check`, `run_phase`.
- **Признаки:** G0 = v3 на d0 + F v3 (`features_r2_v3.parquet`; SHA сверяется с `r2_F_features.yaml`).
- **Контракт драйвера S (PR #184) — уже сделан, после записи задания на передачу:**
  - `r2_roli.py run --designs --h --seeds --steps --workers [--row-frac --row-seed] [--tickers]` и `leak --designs --h`;
  - `r2_resid_grid.py run … [--point pNN]` и `leak`;
  - нога берётся из каталога дизайна (L пишет `leg[__тег]_h{h}_lnall.parquet`); ряды — `B` / `B__<тег>` (варианты ноги L), при подвыборке — `B_rf50…`; у сетки — `BG…`;
  - cross_fit — через `r2_nogi.fit_predict` (оценщик L);
  - точка LHS берётся из `boost.grid_point` (schema_v2): p01…p60, p61_tekushchaya = текущая;
  - LHS драйвера (`driver_v1/D_lhs60.yaml`) совпадает с `r2_B_lhs60.csv` поэлементно.
- **`scripts/analysis/r2_resid_grid.py`:** LHS-60 — `models/designs/r2/r2_B_lhs60.csv`, SHA 8263c0da…; `core_fn`, `run`, `leak`.
- **Контроли и дым** (`data/regen/R2/B/`):
  - движок = R1 D12_h1_A_d812 (шаги 1–3, все тикеры), Δ = 0 — `control_B.csv`;
  - дым: 54 дизайна × 3 h — `smoke_B.csv`; через `run_phase` — `core_smoke_B.csv`;
  - **дым по контракту драйвера** — `drv_smoke`, 3 тикера × шаги 1–3, кэш `R2/B/driver_smoke_cache`, все коды 0 (`driver_smoke_B.csv`). Пройдено: B run h = 5 / 22 / 1, BG run (p01, p61), leak 6/6 на каждом h;
  - сдвиг: 54 из 54 рядов argmax = 0 (`shift_B.csv`);
  - детекторы: ядра — 84 из 84 (`leak_core_B.csv`), cross_fit — 15 из 15 (`leak_xfit_B.csv`), с порчей ноги — 28 из 28 (`leak_B.csv`).

## Что не сделано
- **`r2_dataset_cache`.** Бинарные Dataset на `/Volumes/RAMcache` не подключены, Dataset/QuantileDMatrix строится в воркере. Сложность: X зависит от дизайна (столбец leg_ln, метка роли), поэтому ключ кэша S «шаг, признаки, h, цель, подвыборка» не покрывает ногу. ❓ к S: кэшировать только матрицу признаков, а столбец ноги и метку добавлять при обучении.
- **`schema_v2`.** Модуль валидирует через `r2_core.validate` (это уже v2). Свои ключи `full()` проверяет по `schema_v1.yaml`: листья v1 в v2 не менялись. `grid_point` читается из дизайна.
- **Настройки владельца 16:25 — проверить, что драйвер их передаёт:**
  - скрининг — 1 seed, LGBM, 50% строк: lib и seed задаёт дизайн, row-frac — аргумент `run`;
  - уровень 2 — 3 seed;
  - финалисты — 10 seed, `lib = both_avg`;
  - ≤ 300 деревьев и patience 30 — у `grid = resid_small_v1` и `early_stop = qlike_total` (300 / 30 зашиты, yaml `lhs.nmax` / `patience`);
  - `early_stop = native` — ≤ 1000 деревьев и остановка по val-году, как в R1. ❓ Нужно ли и там 300 и qlike_total, если так не скажет дизайн? Умолчание: дизайн решает.
- **Боевой счёт** не запускался: только дым на 3 тикерах.
- **Тестовые прогнозы** (`test_locked/R2/B/`) — только после ворот E, сырые, только запись. Код шагов 7–10 не написан.

## Особенности и ❓ (подробно — журнал)
- Схема читается `yaml.safe_load`, off/on становятся False/True; нормализация — `onoff()` (ошибка clip = 0 была найдена и исправлена).
- quarterly_warm + gamma-офсет в LightGBM — NotImplementedError (init_model затирает init_score).
- HEAVY + cross_fit: `fit_predict` L требует обучения с начала истории тикера, поэтому будет ValueError (❓ L / O).
- Native-остановка (R1) по построению видит val-год. Детектор ядра поэтому прогоняется на qlike_total, native проверяется срезом 1 января (`leak_leg`).

## Запреты
- Ничего за годы ≥ 2022 не читать. `test_locked` читает только A.
- Не читать таблицы и отчёты теста: `docs/tables/*_test*`, `docs/tables/dlya_teksta/`, отчёты теста гл. 3 и 4, `proverka_laga_harj`, `zhurnal_pervogo_progona_gl3`, `final_test_run.log`.
- Не читать план `docs/zadaniya/2026-10-25_pereschet.md` и постановку владельца `2026-10-26_r2_klastery_vladelec.md`: в них числа теста.
- Не менять `final3.py`, `gibrid_v17.py`, `gibrid_designs.py`, `final_test.py`, `harx.py`, `features_pit_v3`.
- Без `sed -i`, heredoc, `python3 -c` для файлов, `rm -rf`, `pkill -f`; `.env` не читать; сети нет (кроме `git push` и `gh pr`).
