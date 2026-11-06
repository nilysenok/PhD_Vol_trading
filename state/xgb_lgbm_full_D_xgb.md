# Передача: волна D, поток XGB (Ж-206)

Статус на момент записи: код и тесты готовы, GATES_D_xgb.md закоммичен дирижёром. Счёт не начат. Отступления 1–4 приняты дирижёром.

## Файлы (владение)
- `docs/otchety/xgb_lgbm_full_v1/xgboost/GATES_D_xgb.md` (правила Q1–Q9, гейты G1–G12, EVIDENCE: pending).
- `models_v2/s6/xgb/d_space.py, d_fit.py, d_calc.py, d_comp.py, d_regime.py, d_rows.py, d_stages.py, d_stages2.py, runxd.py`.
- `tests/test_v2_xl_xgbd_{leak,mut,fit}.py` (34 passed на момент записи).
- Данные: `data/regen/xgb_lgbm_full_v1/xgboost/d/` (кэш, `rows_D.json`).

## Стадии (строго по порядку, по одной, через run_in_background; не `&`, не heredoc, не `python3 -c`; процессов ≤ 4)
Команда: `python3 -W ignore models_v2/s6/xgb/runxd.py --stage <имя>`
1. `control` (G1): расширяющееся окно D = кэш C побитно; crit_pen 0,20113541 (ctrl_f4) и 0,19692134 (x2_best_j). Пишет `CONTROL_D.json`. При провале — стоп, разбор.
2. `d1` (G4): 12 trials, `WINDOW_AVAILABILITY_D.csv`, строки-опоры.
3. `d1sel` (G5): 2 строки `*_annual_window`, trials не добавляет.
4. `d2` (G6): ≤ 4 trials, `FREQ_DECISION_D.json`.
5. `r1` (G7): 4 trials.
6. `r4` (G8): 4 trials.
7. `tables` (G9): `TEMPORAL_VALIDATION.csv`, `REGIME_PERFORMANCE_D.csv`, синхронизация `CHECKPOINT.json` (used.temporal_regime ≤ 30; план 24).

После счёта: G10 `python3 -m pytest tests/test_v2_xl_xgb*.py -q`, G11 (git status: прежнее не тронуто), G12 `README_D.md` (окна в спокойном/стрессе, влияние 2022, окупается ли ежедневное, R1/R4, отступления до выводов, расход trials), заполнить EVIDENCE в GATES, строка дирижёру. Не коммитить.

## Отступления (приняты)
1. R1/R4 используют stress_RV, не Г4 (Г4 только в срезах). 2. Seed-контролей нет. 3. Веса в строках: 3 года = 756, 5 лет = 1260. 4. Колонки 2017–2021 для x2_best_j — потолок.

## Состояние
Стадии выполнено: 0 из 7. Регрессия G10 в фоне запускалась, результат не получен.
