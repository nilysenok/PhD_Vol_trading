# REPRODUCTION — Ж-212 (SBER h=1)

Порядок (по одной команде, `run_in_background`), из корня репозитория:

- `python3 -W ignore -m models_v2.s8.run_hs --stage ctrl`
- `python3 -W ignore -m models_v2.s8.run_hs --stage table`
- `python3 -W ignore -m models_v2.s8.run_hs --stage diag`
- `python3 -W ignore -m models_v2.s8.run_hs --stage q8`
- `python3 -W ignore -m models_v2.s8.run_hs --stage ledger`
- `python3 -W ignore -m models_v2.s8.run_hs --stage base`
- `python3 -W ignore -m models_v2.s8.run_hs --stage b`
- `python3 -W ignore -m models_v2.s8.run_hs --stage c`
- `python3 -W ignore -m models_v2.s8.run_hs --stage d`
- `python3 -W ignore -m models_v2.s8.run_hs --stage e`
- `python3 -W ignore -m models_v2.s8.run_hs --stage f`
- `python3 -W ignore -m models_v2.s8.run_hs --stage g`
- `python3 -W ignore -m models_v2.s8.run_hs --stage final`

Тесты: `python3 -W ignore -m pytest tests/test_v2_hs_*.py -q`. Кэш рядов — `data/regen/hybrid_struct_v1/*_series.parquet` (ключ — имя конфигурации; удалить файл — пересчёт).
Входы: `data/regen/xgb_lgbm_full_v1/e/components.parquet` (SHA-256 в `CONTROL_VALUES.csv`), `data/iss_check/index_10m/IMOEX.parquet`, `data/fastbt/s9/rvi_1850.parquet`.
Детерминизм: seed 42, num_threads=1 (LGBM), nthread=1 (XGB), spawn-пул ≤ 4.
