# REPRODUCTION — волна E (Ж-206 + Ж-210)

Порядок (по одной, процессов ≤ 4): 
```
python3 -W ignore models_v2/s6/e/run_e.py --stage comp
python3 -W ignore models_v2/s6/e/run_e.py --stage ex0
python3 -W ignore models_v2/s6/e/run_e.py --stage ex1
python3 -W ignore models_v2/s6/e/run_e.py --stage a
python3 -W ignore models_v2/s6/e/run_e.py --stage b
python3 -W ignore models_v2/s6/e/run_e.py --stage c
python3 -W ignore models_v2/s6/e/run_e.py --stage seeds
python3 -W ignore models_v2/s6/e/run_e.py --stage e3
python3 -W ignore models_v2/s6/e/run_e.py --stage e4
python3 -W ignore models_v2/s6/e/run_e.py --stage e5
```
Кэши: `data/regen/xgb_lgbm_full_v1/e/` (ключ = код + вход + seed). Проверка: `python3 -m pytest tests/test_v2_xl_e_*.py -q`.
Ожидаемые значения (EX0): HAR-J(б) 0,2337; ctrl_f4 0,212538; x2_best_j 0,211295; C0_inh_F4 0,2117; Б-2 0,2205; LGBM-аналог 0,2223; c2_h3_roll_f4 0,2098 (допуск 5e-5). Детерминизм: seed 42–46, `deterministic=True`; B-ряд пересчитывается в `--stage e4` побитно.
