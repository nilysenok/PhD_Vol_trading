# GATES Ж-194 (h = 5, 22) — закрыто 5 из 5

Все CHECK выполнены из корня репозитория, `PYTHONPATH=scripts/analysis`.

- G1: CHECK `python3 -m pytest -q tests/test_gl3_zh194.py` · EXPECT: все прошли (закрытое окно h=5 и 22, режим/z дня d+1, PIT логистики и Г6, вес стресса 0, (а)/(б) с равными экспертами, Г6 на константе и v̂ ≥ 0) · EVIDENCE: `11 passed` (вместе с `tests/test_gl3_zh190_stage23.py` — 30 passed) — закрыт.
- G2: CHECK `python3 scripts/analysis/gl3_zh194.py verify` (блок G2) · EXPECT: пересобранные H*_h5 (А-2 с Г-1 c=1) и цепочка Bstar, G1, G2, G3, Gstar, A2_* совпадают с этапом 3, ≤ 1e-9 отн. · EVIDENCE: `G2 max_rel_diff 0.0` (h5: 11 рядов, h22: 4 ряда, `verify.json`) — закрыт.
- G3: CHECK то же, блок G3 · EXPECT: все fit h ∈ {5, 22}, max_target_end < cutoff < 2022-01-01 · EVIDENCE: `events 936, bad 0, horizons [5, 22]` — закрыт.
- G4: CHECK то же, блок G4 · EXPECT: независимый пересчёт метрик из parquet ≤ 1e-11; float32, конечны, > 0, date < 2022, покрытие всех строк эталона dev · EVIDENCE: `max_abs_diff 2.2e-16, values_compared 510` — закрыт.
- G5: CHECK то же, блок G5 + файлы `rows_h5.csv` (30), `rows_h22.csv` (21), `summary.csv`, `selection.json`, `attempt_counts.json` · EXPECT: все строки «Цели», колонка `cel`, выборы в selection.json · EVIDENCE: `rows {5: 30, 22: 21}, ok true` — закрыт.
