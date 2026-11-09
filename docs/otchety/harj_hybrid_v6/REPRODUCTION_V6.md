# REPRODUCTION_V6 — воспроизведение программы HAR-J Hybrid V6 (Ж-227), SBER, h = 1

Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок: 0 (аудит, пороги) → 1a (A0) → 1b (A1–A3, DX-A) → 2 (C1–C4) → 3 (E1–E3) → 4 (D1–D3, K9) → 5 (CB1, F1, F2) → 6 (A0g) → FIN (итог). Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы пишутся в `data/regen/harj_hybrid_v6/`; FIN прогнозов не считает и конфигурации не расходует.

```
python3 -W ignore -m models_v2.s13.run_v6 --stage 0
python3 -W ignore -m models_v2.s13.run_v6 --stage 0 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 1a
python3 -W ignore -m models_v2.s13.run_v6 --stage 1a --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 1b
python3 -W ignore -m models_v2.s13.run_v6 --stage 1b --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 2
python3 -W ignore -m models_v2.s13.run_v6 --stage 2 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 3
python3 -W ignore -m models_v2.s13.run_v6 --stage 3 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 4
python3 -W ignore -m models_v2.s13.run_v6 --stage 4 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 5
python3 -W ignore -m models_v2.s13.run_v6 --stage 5 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage 6
python3 -W ignore -m models_v2.s13.run_v6 --stage 6 --check
python3 -W ignore -m models_v2.s13.run_v6 --stage FIN
python3 -W ignore -m models_v2.s13.run_v6 --stage FIN --check
```

Тесты: `python3 -W ignore -m pytest tests/test_v2_v5_*.py tests/test_v2_v6_*.py -q -p no:cacheprovider`.

Входы (только чтение): сохранённые прогнозы прежних программ SBER и M10-бары SBER (`data/iss_check/bars_10m_full/SBER.parquet`). Выходы: `data/regen/harj_hybrid_v6/`, `docs/otchety/harj_hybrid_v6/` (таблицы, отчёты листов, `CHECKPOINT.json`, `ALL_TRIALS.csv`), `docs/otchety/HARJ_HYBRID_SBER_V6_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.
