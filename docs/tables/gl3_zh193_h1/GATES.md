# GATES — Ж-193 (h=1 максимум), 05.11.2026

CHECK команды запускаются из корня репозитория с `PYTHONPATH=scripts/analysis`.

## G1 тесты
- CHECK: `python3 -m pytest -q tests/test_gl3_zh193_h1.py`
- EXPECT: все прошли (PIT, закрытое окно h=1 и мутация лага 0, мутация «сдвиг вперёд» признака Б1, чтение баров ≥ 2022 падает, известные ответы RV/BV/jump, веса Б4, монотонный вектор Б2, режимные веса Г4, Г5, Г6, мутация внутривыборочного ряда Г4)
- EVIDENCE: `16 passed` (тест `test_g4_calm_and_stress_series_of_year_are_out_of_sample` падал до правки: `assert 0.6175 < 1e-09`, прошёл после правки `dest[va]`)

## G2 ledger
- CHECK: `python3 scripts/analysis/gl3_zh193_h1_verify.py` (блок G2)
- EXPECT: все fit h = 1, `max_target_end < cutoff < 2022-01-01`, bad = 0
- EVIDENCE: `events 13871, bad 0, horizons [1], ok true` (`verify.json`)

## G3 независимый пересчёт
- CHECK: тот же скрипт (блок G3), собственный numpy, эталон dev h=1 `data/regen/gl3_r3/dev/predictions_h1.parquet`
- EXPECT: расхождение ≤ 1e-11; прогнозы float32, конечны, > 0, date < 2022, покрывают все строки эталона
- EVIDENCE: `max_abs_diff 2.2e-16, values_compared 380, series 38, ok true`; воспроизведение этапа 1: `max_abs_dlog_cat 0.0, max_abs_dlog_bstar 0.0`

## G4 строки и выборы
- CHECK: тот же скрипт (блок G4)
- EXPECT: все строки «Цели» в `summary.csv`; `selection.json`, `attempt_counts.json` существуют
- EVIDENCE: `rows 38, missing [], ok true`; файлы `selection.json`, `attempt_counts.json` записаны
