# Передача — волна B, поток LGBM (Ж-206), 06.11.2026 ~00:55. Статус: ПЕРЕДАЧА (счёт идёт/не начат)

Не коммитить. Процессов <= 4. Скрипты только файлами. Гейты: `docs/otchety/xgb_lgbm_full_v1/lightgbm/GATES_B_lgbm.md` (B1–B4 тесты 30 passed, EVIDENCE ещё pending).

## Сделано
Код `models_v2/s6/lgbm/` (grid, fitl, job, crit, tab, world, cfg, st_lb0, st_cfgs, st_run, st_arch, run_b, print_grid), тесты `tests/test_v2_xl_lgbm_{grid,loss,leak,limits}.py`. Кэш/результаты: `data/regen/xgb_lgbm_full_v1/lightgbm/B/` (fc_*.parquet, st_*.json, lb0_folds.parquet, feat_registry.json), таблицы — в `docs/otchety/xgb_lgbm_full_v1/lightgbm/`.

## Порядок счёта (каждая стадия возобновляема, кэш по ключу trial+код)
`python3 models_v2/s6/lgbm/run_b.py --stage lb0` -> loss -> feat1 -> feat2 -> abl -> arch. Стоп-условие: lb0 печатает max отн. разность inherited vs fc_lgbm_b2 (> 1e-9 — стоп и доклад).

## Осталось
Все стадии счёта, EVIDENCE в GATES (B5–B10), ответ дирижёру <= 500 слов (B11). Не менять код после начала счёта без записи ↺ (ключ кэша включает sha кода — смена кода пересчитает всё).
