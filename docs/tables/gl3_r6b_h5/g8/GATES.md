# GATES — Г8 (раунд 6б, h=5), kodir, 05.11.2026 (v2: смещение HAR-J(б) PIT, расширяющееся окно)

Запуск: `PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_g8.py run` (один раз), затем `... audit`.
Версия v1 (обучение с 2016-01-01, смещение — архивный har_corr) сохранена как `*_v1_okno2016*` (таблица, attempt_counts, четыре parquet); её события остались в `fit_events.jsonl` (компоненты G8a, G8b). Технический вариант, в выборе не участвует. Основная версия — компоненты G8a_v2, G8b_v2, HARJ_b.

## G1 — тесты проходят, мутации ловятся
CHECK: python3 -m pytest -q tests/test_gl3_r6b_g8.py
EXPECT: все прошли
EVIDENCE: `8 passed`. Мутации: (1) конец цели в ES-валидации — test_mutation_end_inside_validation_is_caught; (2) перемешанные тикеры c_i и (3) λ = 0 — test_shrink_known_answer_and_mutations, test_g8b_ticker_shuffle_and_lambda_zero_change_forecast; (4) смещение HAR-J(б) с концом цели ≥ cut — test_harj_bases_pit_and_mutation_future_target_caught.

## G2 — ledger: h = 5, max_target_end < cutoff < 2022-01-01; 85 fit Г8а_v2 (17 × 5) + 85 ES
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_g8.py audit
EXPECT: AUDIT OK ... 85 ticker-years
EVIDENCE: `AUDIT OK 367 events 85 ticker-years` (367 = v1 181 + 6 HARJ_b + 170 Г8а_v2 + 10 Г8б_v2); то же 85+85 проверено и для v1.

## G3 — параметры в ledger = У19 (max_depth 3, min_data_in_leaf 100, lambda_l2 10)
CHECK: тот же audit (по каждому событию v1 и v2)
EXPECT: без AssertionError
EVIDENCE: `AUDIT OK`

## G4 — g8_candidates.csv: 4 строки; независимый пересчёт из parquet ≤ 1e−11
CHECK: тот же audit
EXPECT: AUDIT OK
EVIDENCE: `AUDIT OK`; строки Г8а, Г8б_λ126, Г8б_λ504, Г8б_LOYO

## G5 — прогнозы: date < 2022, конечны, > 0, покрывают все 21386 строк эталона dev
CHECK: тот же audit
EXPECT: без AssertionError
EVIDENCE: `AUDIT OK`

## G6 — features_top25.json записан раньше первого dev-fit
CHECK: тот же audit
EXPECT: без AssertionError
EVIDENCE: `AUDIT OK`

## G7 — смещение HAR-J(б) PIT на val-году = архивный har_corr (≤ 1e-8 отн.)
CHECK: тот же audit (harj_check) и поле max_rel_vs_har_corr в событиях HARJ_b
EXPECT: ≤ 1e-8 для каждого года 2016–2021
EVIDENCE: 1,4e-14 (2016), 2,5e-14 (2017), 3,2e-14 (2018), 1,5e-14 (2019), 1,9e-14 (2020), 2,0e-14 (2021); n обучения 6586 … 28006, max конец цели < cut. Схема: ln-OLS на 8 регрессорах (clip 1e-10), обучение год < vy и конец цели < vy, s² sm = var ошибок 3 прошлых val-лет с эмбарго, κ = 0,5; шаг 2016 без поправки (как архив).

## Реестр
закрыто 7 из 7; ABANDON: 0; pending: 0
