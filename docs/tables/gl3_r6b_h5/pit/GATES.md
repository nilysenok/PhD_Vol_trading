# GATES — глава 3, раунд 6б, п. 2б: PIT-пересборка входов стекинга и пересчёт Г3/Г5/Г6/Г7а, kodir, 05.11.2026

Уточнение дирижёра (аудит 2а): HAR-J(б), HAR-CJ _emb_d0, K9, K10 — PIT-чистые, используются как есть. Пересобираются: линейный HAR-X⁺ (HXP) и FIN как вход стекинга (FIN_PIT = D12_h5, замороженные параметры warmup 2016, ES по purge/embargo, годовой refit, s² sm; G1 variant final_fixed без категории тикера). ITER исключён из PIT-версий.

## G1 — тесты, включая две мутации
CHECK: PYTHONPATH=scripts/analysis python3 -m pytest -q tests/test_gl3_r6b_pit.py
EXPECT: все прошли (test_mutation_end_ge_cut_in_training_is_caught, test_s2_ignores_targets_ending_at_or_after_cut_and_mutation_is_caught)
EVIDENCE: 14 passed (rc 0)

## G2 — контроль воспроизведения (архив, собранный нашим кодом, = dev_candidates.csv раунда 6)
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_pit.py --check g2
EXPECT: 5 строк, ok=True, max_abs_diff <= 1e-9
EVIDENCE: G3_annual 5.6e-17, G3_quarterly 2.2e-16, G5 5.6e-17, G6 2.2e-16, G7a_BEST 8.3e-17 — все True (rc 0)

## G3 — ledger: h=5, max_target_end < cutoff < 2022-01-01
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_pit.py --check g3
EXPECT: bad 0
EVIDENCE: events 506 bad 0 (rc 0)

## G4 — archive_vs_pit.csv: 10 строк, leak заполнено
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_pit.py --check g4
EXPECT: 10 0
EVIDENCE: 10 0 (rc 0)

## G5 — сохранённые прогнозы: date < 2022, конечны, > 0, покрывают эталон dev 2017–2021
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_pit.py --check g5
EXPECT: files N ok
EVIDENCE: files 12 ok (rc 0)

## ОТСТУПЛЕНИЕ (не гейт)
↺ ITER (D26/x6_h5) не пересобран: честная пересборка требует всей цепочки h=1 (hyb₁ из L2_h1 с Optuna/ES на val-году); по уточнению дирижёра исключён из PIT-версий Г3/Г5/Г6 (ключи PIT: FIN_PIT, HAR, CJ, HXP_PIT, K9). Дополнительно (диагностика, не основная таблица) посчитан вариант «PIT + ITER архивный» — archive_vs_pit_iter_variant.csv.

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
