# Воспроизведение SBER h=1 deep (Ж-204)

Окружение: Python 3.9; numpy 2.0.2, pandas 2.3.3, xgboost 2.1.4, lightgbm 4.6.0, optuna 4.7.0, matplotlib 3.9.4; `nthread/num_threads = 1`, seed 42 (seeds 43–46 — устойчивость). Кэш — с ключом = хэш входов и кода (`data/regen/v2_sber_h1_deep_v1/{b1,b2,c,d}`).

| Волна | Команда |
|---|---|
| A | `python3 models_v2/s5/run_a.py --stage all` |
| B1 | `python3 models_v2/s5/b1/run_b1.py --stage all` |
| B2 | `python3 models_v2/s5/b2/run_b2.py --stage all` |
| C | `python3 models_v2/s5/c/run_c.py --stage all` |
| D | `python3 models_v2/s5/d/run_d.py --stage all` (по стадиям: control, comp, d1, robust, d2, report) |
| Тесты | `python3 -m pytest tests/test_v2_*.py -q` |

Входы: `data/iss_check/{bars_10m_full,index_10m}`, `data/fastbt/s9/rvi_1850.parquet`, `data/regen/v2_sber_h1*` (хэши — `MODEL_SPECIFICATIONS.json`, `BASELINE_MANIFEST.json`). Контроль: Б-2 0,220486502764, HAR-J(б) 0,233662115694, 2271 дата. Лучший ансамбль D1: `e3_roll`. Перепроверка D: `python3 models_v2/s5/d/run_d.py --stage control` (код 1 при расхождении) и `pytest tests/test_v2_deep_d_*.py`.
