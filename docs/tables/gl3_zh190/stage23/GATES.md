# GATES — Ж-190 этапы 2–3 (h = 1, 5, 22), kodir, 05.11.2026

## G1 — один набор тестов на модуль проходит (PIT, закрытые окна Г-3/А-3, сдвиг вперёд, c = 0 Г-1, w = 0 Г-3, известные ответы)
CHECK: python3 -m pytest -q tests/test_gl3_zh190_stage23.py
EXPECT: все прошли
EVIDENCE: 19 passed, 20 warnings (код возврата 0). Мутации в тестах: конец цели ≥ cut в ledger; purge по дате строки; окно s = d − h + 1 (Г-3 и А-3); маска весов А-1/А-2 с концом цели до cut + 20 дней; режим дня d + 1 (RVI_LEAD = 1); знак f у Г-1.

## G2 — ledger: все fit h ∈ {1, 5, 22}, max_target_end < cutoff < 2022-01-01
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_stage23_verify.py   (блок G2)
EXPECT: bad = 0, horizons = [1, 5, 22]
EVIDENCE: events 984, bad 0, horizons [1, 5, 22], ok true (verify.json)

## G3 — независимый пересчёт метрик из parquet ≤ 1e−11; прогнозы конечны, > 0, date < 2022, покрывают все строки эталона dev
CHECK: тот же запуск (блок G3 и hari)
EXPECT: max_abs_diff ≤ 1e-11; числа тикеров лучше эталона совпадают; HAR_i из файлов λ = HARi_LOYO
EVIDENCE: max_abs_diff 2.2e-16 по 780 значениям; hari max_rel_diff 0.0; coverage-assert и float32 / > 0 / < 2022 пройдены для всех рядов

## G4 — все строки «Цели» по каждому h; selection.json; attempt_counts.json
CHECK: тот же запуск (блок G4)
EXPECT: 26 строк на h, ключи выбора на месте
EVIDENCE: rows 26 / 26 / 26, ok true; selection.json (λ LOYO, c-выборы Г-1/Г-2, Г*, А-2, А-3, H*), attempt_counts.json (984 события ledger, 30 заданий Г-1/Г-2, 23 ряда на h)

## ABANDON
Нет.

## Реестр
закрыто 4 из 4; ABANDON: 0; pending: 0
