## R-а (реверс раунд 1) — статус: код готов, test1 не считан

Скрипт `scripts/revers/r1_a.py --period {train4,test1}`. train4 воспроизводит `docs/tables/dve_metriki/a_strogaya_yakor.csv` с max|Δ| = 8.9e-16 (допуск 1e-9). Для A и A_strogaya: Sharpe (ddof=1, √252) и mSR, полный ряд, без 10 худших / лучших дней, стоп 2σ, размер 1/σ. Таблица test1 — `docs/tables/revers_r1/a_dve_metriki_test1.csv` (после снятия замка и слияния R-P).

Не сделано: TEST_C и D поверх A — на train4 s9 σ-источники не читает, их σ̂ 2022–2024 даёт R-P; подключить после слияния R-P (интерфейс неизвестен).
