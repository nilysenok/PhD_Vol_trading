# GATES — Э1-а, тестировщик движка перебора (Ж-034/035), ветка sweep-tests

Тесты: `tests/test_sweep_engine.py` (оценщик, синтетика), `test_sweep_drivers.py` (стык: кэш, ledger/N→DSR, контракт),
`test_sweep_modul.py` (модуль владельца `sweep_drivers.py` + сквозной оценщик→oat_sweep), `test_sweep_mutacii.py` и
`test_sweep_modul_mut.py` (подложенные ошибки в памяти). Код `fastbt/` не менялся. v1.1 — `@pytest.mark.v11`.
Запуск везде: `/usr/bin/python3 -m pytest tests/test_sweep_*.py -q -W ignore -k <метка>` (без `-n`).
Мутация на гейте ловится тестами `test_mut*`, входящими в метку; детектор на правильном коде проверяется парным тестом.

## G1 — параметр изменён → результат изменился (сигнальный, PnL, OAT модуля)
CHECK: ... -k param
EXPECT: passed, 0 failed
EVIDENCE: 18 passed, 52 deselected

## G2 — validation не участвует в целевой; 2020 подмешан в train → результат меняется
CHECK: ... -k validation
EXPECT: passed, 0 failed
EVIDENCE: 4 passed, 66 deselected

## G3 — границы периодов (настройка/контроль/validation не пересекаются, train до 01.01.2020)
CHECK: ... -k granitsy
EXPECT: passed, 0 failed
EVIDENCE: 1 passed, 69 deselected (+ test_mut_04/06 в -k mut)

## G4 — комиссия с обеих сторон
CHECK: ... -k komissiya
EXPECT: passed, 0 failed
EVIDENCE: 7 passed, 63 deselected

## G5 — PBO на случайных данных ≈ 0,5 (v11, допуск ±0,10 объявлен до счёта)
CHECK: ... -k pbo
EXPECT: passed, 0 failed
EVIDENCE: 4 passed, 66 deselected

## G6 — DSR при N=1 растёт (мутация № 14)
CHECK: ... -k dsr
EXPECT: passed, 0 failed
EVIDENCE: 3 passed, 67 deselected

## G7 — Д5: ключ кэша меняется при подмене ряда, кода, параметра сигнала
CHECK: ... -k klyuch
EXPECT: passed, 0 failed
EVIDENCE: 9 passed, 61 deselected

## G8 — кэш сигналов отделён от PnL
CHECK: ... -k kesh_pnl
EXPECT: passed, 0 failed
EVIDENCE: 1 passed, 69 deselected

## G9 — OAT: число проб по поведению модуля (1 + Σ|сетка_i|), остальные фиксированы
CHECK: ... -k oat
EXPECT: passed, 0 failed
EVIDENCE: 4 passed, 66 deselected

## G10 — плато = среднее по соседям (в модуле — включая саму точку), не на пике
CHECK: ... -k plato
EXPECT: passed, 0 failed
EVIDENCE: 3 passed, 67 deselected

## G11 — Шарп ddof=1 (мутация № 12); σ в objective_stable — ddof=0, тест текущего поведения (вопрос владельцу 27.10)
CHECK: ... -k ddof
EXPECT: passed, 0 failed; 1 xfailed
EVIDENCE: 3 passed, 66 deselected, 1 xfailed

## G12 — целевая = среднее годовых Шарпов − 0,5·σ (λ=0,5 и штраф; ddof — см. G11)
CHECK: ... -k tselevaya
EXPECT: passed, 0 failed; 1 xfailed
EVIDENCE: 1 passed, 68 deselected, 1 xfailed

## G13 — ≥ 2022 не читается
CHECK: ... -k 2022
EXPECT: passed, 0 failed
EVIDENCE: 2 passed, 68 deselected

## G14 — повторное открытие validation: запись обязательна, N×2
CHECK: ... -k otkrytie
EXPECT: passed, 0 failed
EVIDENCE: 2 passed, 68 deselected

## G15 — файлы ≤ 200 строк, py_compile
CHECK: wc -l tests/test_sweep_{engine,drivers,modul,modul_mut,mutacii}.py; /usr/bin/python3 -m py_compile tests/test_sweep_*.py
EXPECT: каждый ≤ 200, код 0
EVIDENCE: 196 / 96 / 200 / 103 / 162; py_compile ok

## ABANDON

## Реестр
закрыто 15 из 15; ABANDON: 0 (остались 2 xfail strict v11: Моррис σ, Optuna n_jobs)
