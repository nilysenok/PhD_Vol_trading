# REPRODUCTION_V4 — воспроизведение программы HAR-J Hybrid V4

Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии, процессов 1. Порядок — порядок этапов (audit → A → B → Ca → Cb → E → DX → CS → NTP → CLD → RQC → H → I → FIN); стадия без `--check` ПИШЕТ выходы (кэши, таблицы, отчёты, реестр), `--check` только читает (стадии audit, A, B, Ca, Cb, E, CS, NTP, CLD, RQC — в песочнице `v4_sandbox`; DX, H, I, FIN — чтение с диска). Повторная запись идемпотентна (ключ кэша = хэш входов и кода, Д5).

```
python3 -W ignore -m models_v2.s11.run_v4 --stage audit
python3 -W ignore -m models_v2.s11.run_v4 --stage audit --check
python3 -W ignore -m models_v2.s11.run_v4 --stage A
python3 -W ignore -m models_v2.s11.run_v4 --stage A --check
python3 -W ignore -m models_v2.s11.run_v4 --stage B
python3 -W ignore -m models_v2.s11.run_v4 --stage B --check
python3 -W ignore -m models_v2.s11.run_v4 --stage Ca
python3 -W ignore -m models_v2.s11.run_v4 --stage Ca --check
python3 -W ignore -m models_v2.s11.run_v4 --stage Cb
python3 -W ignore -m models_v2.s11.run_v4 --stage Cb --check
python3 -W ignore -m models_v2.s11.run_v4 --stage E
python3 -W ignore -m models_v2.s11.run_v4 --stage E --check
python3 -W ignore -m models_v2.s11.run_v4 --stage CS
python3 -W ignore -m models_v2.s11.run_v4 --stage CS --check
python3 -W ignore -m models_v2.s11.run_v4 --stage NTP
python3 -W ignore -m models_v2.s11.run_v4 --stage NTP --check
python3 -W ignore -m models_v2.s11.run_v4 --stage CLD
python3 -W ignore -m models_v2.s11.run_v4 --stage CLD --check
python3 -W ignore -m models_v2.s11.run_v4 --stage RQC
python3 -W ignore -m models_v2.s11.run_v4 --stage RQC --check
python3 -W ignore -m models_v2.s11.run_v4 --stage H
python3 -W ignore -m models_v2.s11.run_v4 --stage H --check
python3 -W ignore -m models_v2.s11.run_v4 --stage I
python3 -W ignore -m models_v2.s11.run_v4 --stage I --check
python3 -W ignore -m models_v2.s11.run_v4 --stage FIN
python3 -W ignore -m models_v2.s11.run_v4 --stage FIN --check
python3 -W ignore -m models_v2.s11.run_v4 --stage DX
python3 -W ignore -m models_v2.s11.run_v4 --stage DX --check
```

Тесты: `python3 -W ignore -m pytest tests/test_v2_v4_*.py -q -p no:cacheprovider`.

Входы (только чтение): `data/regen/harj_panel_hybrid` (K1 `har_<T>.parquet`, K2/K3 `k23_<T>.parquet`, D2 `hyb_lgbm_p17.parquet`, `engine_panel_p17.parquet`, `daily_<T>.parquet`), M10-данные проекта. Выходы: `data/regen/harj_hybrid_v4/` (прогнозы 46 конфигураций, кэш), `docs/otchety/harj_hybrid_v4/` (таблицы этапов, `FIN_*.csv`, отчёты), `docs/otchety/HARJ_HYBRID_V4_FINAL.md`. Стадия FIN конфигураций не расходует, parquet не пишет; её `--check` пересобирает `FIN_*.csv` из прогнозов в памяти, сверяет с диском и с таблицами этапов (допуск 1e-9) и проверяет числа итогового документа.
Ключи и хэши: `CHECKPOINT.json` (`FIN_prior_parquet_sha`, `FIN_prior_tables_sha`, `*_cache_key`).
