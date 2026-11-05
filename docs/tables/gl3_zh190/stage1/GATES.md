# GATES — Ж-190 этап 1, потикерный бустинг, h = 1, 5, 22 (агент 1), 05.11.2026

Команды из корня репозитория, `PYTHONPATH=scripts/analysis`. Полный прогон (`gl3_zh190_boost.py all`): 303 с на 8 процессах × 1 поток (старт 09:49 UTC).

## G1 — тесты модуля
CHECK: `python3 -m pytest -q tests/test_gl3_zh190_boost.py -p no:warnings`
EXPECT: все прошли; PIT, purge/embargo h, мутации (сдвиг вперёд, без embargo), градиент и гессиан QLIKE по численной производной, cont: 50..200 деревьев сверх панели
EVIDENCE: `27 passed in 4.07s` (код 0)

## G2 — контроль переиспользования
CHECK: `python3 scripts/analysis/gl3_zh190_boost_checks.py g2` (файл `g2_reuse.json`)
EXPECT: новый тренер в режиме совместимости (K9: max_bin 255, 2 потока, 300/20, без embargo) совпадает с `Engine.boost` K9/K10 на h = 5, отсечки 2016–2021, ≤ 1e-6 отн.
EVIDENCE: `ok: true`; 12 сравнений (mse↔K9, qlike↔K10 × 6 отсечек), `max_rel_diff = 0.0` во всех 12. Сверено с K9 и K10, а не с ногой LGB FIN_PIT: `FIN_PIT.parquet` — среднее уровней LGB и XGB, одна нога не сохранена. (panel, huber1) в K9/K10 нет — им не покрыт.

## G3 — ledger
CHECK: `python3 scripts/analysis/gl3_zh190_boost_checks.py verify` (блок G3, `fit_events.jsonl`)
EXPECT: все fit: h ∈ {1,5,22}, max_target_end < cutoff < 2022-01-01
EVIDENCE: events 14621, bad 0, horizons [1,5,22], ok true

## G4 — независимый пересчёт метрик
CHECK: `... verify` (блок G4: свой numpy-код по `forecasts_h*.parquet`, без gl3_r6b_common; RVI-стресс считается заново из `features_pit_v3`)
EXPECT: расхождение с таблицами ≤ 1e-11; прогнозы конечны, > 0, date < 2022, покрывают все строки эталона dev (2017–2021) по каждому h
EVIDENCE: max_abs_diff 2.2e-16, сравнено 300 значений (10 строк × 3 h × 10 метрик), ok true; матрицы: float32, имена = `feature_names[h]` (rv_d/rv_w/rv_m как ln_*), date ≤ 2021-12-30, без `har_feature`/`ln_HAR_b`

## G5 — полнота таблиц
CHECK: `... verify` (блок G5, `attempt_counts.json`)
EXPECT: 60 отсева, 21 на пять лет, 6 итерации 3, 3 итерации 4, 3 строки Б*_h
EVIDENCE: screen 60, stage1 21, iter3 6, iter4 3, bstar 3; ok true. fit: h=1 — 2068 refit / 6148 inner_ES, h=5 — 886 / 3646, h=22 — 504 / 1224; Optuna-проб 2160; ячеек (h, год, конфигурация) по 86 на h
