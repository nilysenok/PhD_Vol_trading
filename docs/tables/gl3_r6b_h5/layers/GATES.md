# GATES — Г9, Г10, Г11, Г12, Г14_FIN, Г15 (раунд 6б, h=5, волна 2), kodir, 05.11.2026

Запуск: `PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_layers.py run` (один раз; повтор запрещён, У12), затем `... audit`.

## G1 — тесты проходят, мутации ловятся (У13)
CHECK: python3 -m pytest -q tests/test_gl3_r6b_layers.py
EXPECT: все прошли
EVIDENCE: `17 passed`. Мутации: (1) s = d−4 в закрытом окне, Г10/Г14/Г15 — test_closed_window_no_leak_and_mutation_s_equals_d_minus_4[10|14|15] (LAG=4 меняет прогноз; LAG=5 — нет); (2) режим по RVI дня d+1 — test_regime_rvi_uses_only_past_and_mutation_next_day (RVI_LEAD=1); день d не входит в квантиль RV_w — test_regime_rvw_quantile_excludes_day_d; (3) перемешанные тикеры: Г10/Г14 — test_ticker_shuffle_caught_g10_g14, Г15 — test_g15_per_ticker_weights_and_shuffle_caught, Г11 — test_g11_ticker_shuffle_caught; (4) обрезка k снята: Г11 — test_g11_known_answer_clip_and_mutation (K11_CLIP), Г14 — test_g14_clip_and_mutation (K14_CLIP); конец цели ≥ cut в калибровке Г11 — test_g11_no_future_target_in_calibration; LOYO с учётом held-года — test_hard_gate_and_loyo_excludes_held_year.

## G2 — известный ответ на синтетике
CHECK: python3 -m pytest -q tests/test_gl3_r6b_layers.py -k "known_answer"
EXPECT: прошли
EVIDENCE: `3 passed` (в общем прогоне 17 passed): RV ≡ c·σ̂² → k = c (Г10, Г14 при hl 20/120, w .5/.75; QLIKE = 0); Q^hyb = Q^HAR (r₁=2, r₂ из brentq) → w = 0,5 (Г15); Г11: exp(s²/2) по ±s-ошибкам, 1,25 при обрезке.

## G3 — ledger: h = 5, max_target_end < cutoff < 2022-01-01
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_layers.py audit
EXPECT: AUDIT OK ... events
EVIDENCE: `AUDIT OK 24 rows 20 events` (20 = 8 fit логистики Г9/Г12 за 2018–2021 + 9 calibration Г11 (FIN 5 + BEST 4) + 3 fallback: Г9 2017, Г12 2017, Г11_BEST 2017 — у PIT-кандидатов нет истории до 2017).

## G4 — независимый пересчёт метрик из parquet ≤ 1e−11; прогнозы конечны, > 0, date < 2022, float32, покрывают 21386 строк эталона
CHECK: тот же audit
EXPECT: без AssertionError
EVIDENCE: `AUDIT OK 24 rows 20 events` (годовые QLIKE, среднее и stress2020 по каждому из 24 parquet пересчитаны из rv_actual эталона).

## G5 — таблица содержит все строки «Цели»; LOYO-выборы записаны в selection.json
CHECK: audit (len(candidates.csv) == 24, selection.json существует)
EXPECT: AUDIT OK
EVIDENCE: `AUDIT OK`; selection.json: picks (calm_G9, Г9, Г10_FIN/BEST_LOYO, Г12, Г14_FIN_LOYO), best_pit_pick = best_pit_recorded_in_pit_G7a_BEST (совпадают по всем 5 годам: G2_logistic, G3_annual, G2_logistic, G3_quarterly, G3_annual).

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
