# REPRODUCTION — Ж-211, SBER h=1

Из корня репозитория, по одной стадии (процессов ≤ 6; retrain — spawn-пул 4):
```
python3 -W ignore -m models_v2.s7.run_hf --stage reg      # реестр, N_prior
python3 -W ignore -m models_v2.s7.run_hf --stage diag     # диагностика I
python3 -W ignore -m models_v2.s7.run_hf --stage bound    # R1/R2/R3
python3 -W ignore -m models_v2.s7.run_hf --stage inter    # тест режимов (ABANDON при отсутствии)
python3 -W ignore -m models_v2.s7.run_hf --stage weights  # EMA и лимит скорости
python3 -W ignore -m models_v2.s7.run_hf --stage feat     # retrain наборов признаков (≈40 с на ряд)
python3 -W ignore -m models_v2.s7.run_hf --stage final    # тесты, уровни, таблицы, документы
python3 -m pytest tests/test_v2_hf_rules.py tests/test_v2_hf_stages.py tests/test_v2_hyb_leak.py tests/test_v2_xl_e_bfit.py -q
```
Входы: `data/regen/xgb_lgbm_full_v1/e/` (components, b_B1q/B2q_s42, series_a/c), `feats_b2` (v2_sber_h1). Выходы: `docs/otchety/hybrid_final_v1/`, ряды — `data/regen/hybrid_final_v1/`.
Детерминизм: seed 42, блочный бутстрап (20, 5000, seed 42); кэш retrain — по хэшу кода и матрицы признаков.
