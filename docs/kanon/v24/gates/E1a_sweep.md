# GATES — Ж-031 Э1-а, движок перебора fastbt/sweep.py, kodir, 27.10

Мутации в тестах — в памяти (monkeypatch), реальный код не правится. FASTBT_MAIN — своя копия.

## G1 — параметр изменён → результат изменился
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k param_changes
EXPECT: passed, 0 failed
EVIDENCE: pending

## G2 — validation не влияет на выбор: подмешали 2020 в train → целевая меняется; сама целевая не видит 2020–2021
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k validation
EXPECT: passed, 0 failed
EVIDENCE: pending

## G3 — границы настройка/контроль/validation не пересекаются
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k periods
EXPECT: passed, 0 failed
EVIDENCE: pending

## G4 — комиссия списывается с обеих сторон
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k commission
EXPECT: passed, 0 failed
EVIDENCE: pending

## G5 — PBO на случайных данных ≈ 0,5 (допуск ±0,12 при 40 точках, 8 блоков, seed)
CHECK: /usr/bin/python3 -m pytest tests/test_sweep_stat.py -q -k pbo
EXPECT: passed, 0 failed
EVIDENCE: pending

## G6 — DSR при N=1 больше, чем при N>1
CHECK: /usr/bin/python3 -m pytest tests/test_sweep_stat.py -q -k dsr
EXPECT: passed, 0 failed
EVIDENCE: pending

## G7 — Моррис на аналитической функции даёт известные μ*
CHECK: /usr/bin/python3 -m pytest tests/test_sweep_modes.py -q -k morris
EXPECT: passed, 0 failed
EVIDENCE: pending

## G8 — Optuna с фиксированным seed воспроизводится; n_jobs ≤ 8
CHECK: /usr/bin/python3 -m pytest tests/test_sweep_modes.py -q -k optuna
EXPECT: passed, 0 failed
EVIDENCE: pending

## G9 — кэш сигналов: ключ меняется от параметров, ряда и кода; PnL отдельно
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k cache
EXPECT: passed, 0 failed
EVIDENCE: pending

## G10 — режим validation: флаг «validation открыта», повторное открытие без записи и удвоения N запрещено
CHECK: /usr/bin/python3 -m pytest tests/test_sweep.py -q -k reopen
EXPECT: passed, 0 failed
EVIDENCE: pending

## G11 — дымовой прогон OAT, один параметр, S1, train 2018, реальный ряд: OK и время
CHECK: FASTBT_MAIN=$PWD FASTBT_CACHE=<scratchpad>/cache /usr/bin/python3 -m fastbt.sweep_smoke
EXPECT: строка OK и время, числа не печатаются
EVIDENCE: pending

## G12 — файлы ≤ 200 строк, py_compile
CHECK: wc -l fastbt/sweep*.py tests/test_sweep*.py; /usr/bin/python3 -m py_compile fastbt/sweep*.py
EXPECT: каждый ≤ 200, код возврата 0
EVIDENCE: pending

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 0 из 12; ABANDON: 0; pending: 12
