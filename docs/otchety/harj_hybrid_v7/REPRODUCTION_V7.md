# REPRODUCTION_V7 — воспроизведение программы HAR-J Hybrid V7 (Ж-232), SBER, h = 1

Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии. Порядок: 0 (аудит, паспорт M10, последовательности) → 1a (инфраструктура, реестр) → 1b (энкодеры, конфигурации 1–3) → 2 (конфигурации 4–7) → DX (диагностика) → 3 (конфигурация 8) → 4 (конфигурация 10; 11 не считается) → FIN (итог). Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы — `data/regen/harj_hybrid_v7/`; FIN прогнозов не считает, конфигурации не расходует.

```
python3 -W ignore -m models_v2.s14.run_v7 --stage 0
python3 -W ignore -m models_v2.s14.run_v7 --stage 0 --check
python3 -W ignore -m models_v2.s14.run_v7 --stage 1a
python3 -W ignore -m models_v2.s14.run_v7 --stage 1a --check
python3 -W ignore -m models_v2.s14.run_v7 --stage 1b
python3 -W ignore -m models_v2.s14.run_v7 --stage 1b --check
python3 -W ignore -m models_v2.s14.run_v7 --stage 2
python3 -W ignore -m models_v2.s14.run_v7 --stage 2 --check
python3 -W ignore -m models_v2.s14.run_v7 --stage DX
python3 -W ignore -m models_v2.s14.run_v7 --stage DX --check
python3 -W ignore -m models_v2.s14.run_v7 --stage 3
python3 -W ignore -m models_v2.s14.run_v7 --stage 3 --check
python3 -W ignore -m models_v2.s14.run_v7 --stage 4
python3 -W ignore -m models_v2.s14.run_v7 --stage 4 --check
python3 -W ignore -m models_v2.s14.run_v7 --stage FIN
python3 -W ignore -m models_v2.s14.run_v7 --stage FIN --check
```

Тесты: `python3 -W ignore -m pytest tests/test_v2_v7_*.py -q -p no:cacheprovider`.

**Протокол регрессии (G54).** Причина риска: lightgbm и xgboost (V5–V6) и torch (V7) приносят каждый свою библиотеку OpenMP; многопоточный torch после загрузки lightgbm/xgboost роняет процесс сегфолтом (rc 139) на первом softmax. Поэтому `models_v2/s14/__init__.py` закрепляет torch на одном потоке (`set_num_threads(1)`, `set_num_interop_threads(1)`) при импорте пакета; V7 считает в один поток, числа не меняются. С закреплением единый прогон одним процессом проходит: `python3 -m pytest -q -p no:cacheprovider tests/test_v2_v5_*.py tests/test_v2_v6_*.py tests/test_v2_v7_*.py` (860 passed). Тесты V7 можно запускать и отдельным процессом.

Выходы: `docs/otchety/harj_hybrid_v7/` (таблицы, отчёты листов, `FIN_*.csv`, `FIN_WEIGHTS.png`, `CHECKPOINT.json`, `ALL_TRIALS.csv`), `docs/otchety/HARJ_HYBRID_SBER_V7_M10_MOE_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.
