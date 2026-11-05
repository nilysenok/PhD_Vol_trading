# GATES — Ж-190 эталоны и потикерный HAR-J со сжатием (h = 1, 5, 22), kodir (агент 2), 05.11.2026

Команды — из корня репозитория, `PYTHONPATH=scripts/analysis`.

## G1 — тесты HAR_i проходят и ловят подложенные ошибки
CHECK: python3 -m pytest -q tests/test_gl3_zh190_har.py
EXPECT: все прошли; ловятся: коэффициенты по строкам с концом цели >= cut, s² по строкам с концом цели >= cut, перемешанные тикеры, λ=1 = потикерный OLS, λ=0 = общий
EVIDENCE: `9 passed` (rc 0). Мутации внутри тестов: emb=False (β и s²), перемешанные метки тикеров, LOYO со своим годом, record с концом цели >= cut, данные >= 2022.

## G2 — λ = 0 совпадает с сохранённым har_corr (<= 1e-8 отн.) для h = 1, 5, 22
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_har.py audit g2
EXPECT: max_rel <= 1e-8 по каждому h (данные из docs/tables/gl3_zh190/etalony/g2_lam0_h{h}.json)
EVIDENCE: h1 max_rel 2.94e-14 (n 25738); h5 3.20e-14 (n 25670); h22 3.23e-14 (n 25381) — rc 0.

## G3 — FIN_PIT: h5 побайтно файл 6б; h1 ledger с max_target_end < cutoff < 2022-01-01; h22 ≡ HAR-J(б)
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_har.py audit g3
EXPECT: sha h5 = 474e92e5fcf9f0c9…; все fit ledger в порядке; FIN_PIT_h22 = HARJb_h22 побайтно
EVIDENCE: `events 396 bad 0 FIN_h1 refit 30 ES 30`; `FIN_PIT_h5 sha 474e92e5fcf9f0c9`; `FIN_PIT_h22 == HARJb_h22 побайтно` — rc 0.

## G4 — независимый пересчёт метрик из parquet <= 1e-11; прогнозы конечны, > 0, date < 2022, покрывают все строки эталона dev
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_har.py audit g4
EXPECT: AUDIT G4 OK
EVIDENCE: `AUDIT G4 OK` (rc 0; пересчёт: годовые, mean, обе стресс-строки, обе RV/σ̂², число тикеров лучше каждого эталона, все 9 строк × 3 h).

## G5 — три таблицы etalony_h{h}.csv со всеми строками; in-window прогнозы для каждой отсечки и h
CHECK: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_har.py audit g5
EXPECT: AUDIT G5 OK (9 строк в каждой таблице; 6 отсечек in-window на h)
EVIDENCE: `AUDIT G5 OK` (rc 0).

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
