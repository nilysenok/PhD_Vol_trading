# REPRODUCTION_V5 — воспроизведение программы HAR-J Hybrid V5 (Ж-226)

Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок: 0 (аудит, критерий, паспорт) → DX1 (диагностики 1b) → GS (лист 1a) → SP (лист 2) → FIN (итог). Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы GS и SP пишутся в `data/regen/harj_hybrid_v5/` с ключом кэша = хэш входов и кода (Д5); FIN прогнозов не считает, конфигурации не расходует. GS: до GS3–GS5 нужно записать форму score (`--decide-form` после `--cfg GS1`, `--cfg GS2`); `--cfg GSn` считает одну конфигурацию.

```
python3 -W ignore -m models_v2.s12.run_v5 --stage 0
python3 -W ignore -m models_v2.s12.run_v5 --stage 0 --check
python3 -W ignore -m models_v2.s12.run_v5_dx --stage DX1
python3 -W ignore -m models_v2.s12.run_v5_dx --stage DX1 --check
python3 -W ignore -m models_v2.s12.run_v5 --stage GS
python3 -W ignore -m models_v2.s12.run_v5 --stage GS --check
python3 -W ignore -m models_v2.s12.run_v5 --stage SP
python3 -W ignore -m models_v2.s12.run_v5 --stage SP --check
python3 -W ignore -m models_v2.s12.run_v5 --stage FIN
python3 -W ignore -m models_v2.s12.run_v5 --stage FIN --check
```

Отдельные шаги GS: `python3 -W ignore -m models_v2.s12.run_v5 --stage GS --cfg GS1` (и GS2), затем `--stage GS --decide-form`, затем `--cfg GS3`…`GS5`.

Тесты: `python3 -W ignore -m pytest tests/test_v2_v5_*.py -q -p no:cacheprovider`.

Входы (только чтение): прогнозы V4 (`data/regen/harj_hybrid_v4`, `harj_panel_hybrid`), M10-бары 17 бумаг. Выходы: `data/regen/harj_hybrid_v5/`, `docs/otchety/harj_hybrid_v5/` (таблицы, отчёты, `CHECKPOINT.json`, `ALL_TRIALS.csv`), `docs/otchety/HARJ_HYBRID_V5_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.
