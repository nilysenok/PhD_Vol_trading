# GATES — Э1-а, тестировщик движка перебора (Ж-034), ветка sweep-tests

Тесты: `tests/test_sweep_engine.py` (ядро, синтетика), `tests/test_sweep_drivers.py` (режимы, статистика; v1.1 — `@pytest.mark.v11`), `tests/test_sweep_mutacii.py` (подложенные ошибки в памяти). Код `fastbt/` не меняется. Каждый гейт: свойство на правильном коде проходит (`-k eng_...`), мутация роняет проверку (`-k mut_...`).
Запуск везде: `/usr/bin/python3 -m pytest tests/test_sweep_engine.py tests/test_sweep_drivers.py tests/test_sweep_mutacii.py -q -k <метка>`.

## G1 — параметр изменён → результат изменился
CHECK: ... -k "param"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G2 — validation не участвует в целевой; подмешали 2020 в train → результат меняется
CHECK: ... -k "validation_zakryta or podmeshali"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G3 — границы периодов
CHECK: ... -k "granitsy"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G4 — комиссия с обеих сторон
CHECK: ... -k "komissiya"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G5 — PBO на случайных данных ≈ 0,5 (v11)
CHECK: ... -k "pbo"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G6 — DSR при N=1 растёт (мутация № 14)
CHECK: ... -k "dsr"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G7 — Д5: ключ кэша меняется от ряда, кода, параметра сигнала
CHECK: ... -k "klyuch"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G8 — кэш сигналов отделён от PnL
CHECK: ... -k "kesh_pnl"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G9 — OAT: Σ(|сетка_i|−1)+1 точек, остальные фиксированы
CHECK: ... -k "oat"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G10 — плато = среднее по соседям
CHECK: ... -k "plato"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G11 — Шарп ddof=1 (мутация № 12)
CHECK: ... -k "ddof"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G12 — целевая = среднее годовых Шарпов − 0,5·σ по годам
CHECK: ... -k "tselevaya"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G13 — ≥ 2022 не читается вовсе
CHECK: ... -k "2022"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G14 — файлы ≤ 200 строк, py_compile
CHECK: wc -l tests/test_sweep_engine.py tests/test_sweep_drivers.py tests/test_sweep_mutacii.py; /usr/bin/python3 -m py_compile tests/test_sweep_*.py
EXPECT: каждый ≤ 200, код 0
EVIDENCE: pending

## ABANDON

## Реестр
закрыто 0 из 14; ABANDON: 0; pending: 14
