# GATES — К-wf6 лист 1 (ядро CSCV по кварталам), kodir, 29.10

## G1 — 45 кварталов → 10 блоков из целых кварталов, границы в коде; не 45 кварталов — отказ
CHECK: python3 -m pytest tests/test_pbo_wf.py -q -k "bloki"
EXPECT: passed
EVIDENCE: 5 passed, 9 deselected

## G2 — 252 разбиения, половины дополняют друг друга, выбор видит только блоки in-sample
CHECK: python3 -m pytest tests/test_pbo_wf.py -q -k "razbieniya or vidit"
EXPECT: passed
EVIDENCE: 2 passed, 12 deselected

## G3 — PBO на шуме ≈ 0,5; на заложенном сигнале ≈ 0
CHECK: python3 -m pytest tests/test_pbo_wf.py -q -k "shum or signal"
EXPECT: passed
EVIDENCE: 2 passed, 12 deselected

## G4 — граница: пустая лента, нулевой объём блока, nan — отказ/nan, не молчаливый ответ
CHECK: python3 -m pytest tests/test_pbo_wf.py -q -k "gran"
EXPECT: passed
EVIDENCE: 5 passed, 9 deselected

## G5 — py_compile и весь файл тестов
CHECK: python3 -m py_compile fastbt/pbo_wf.py && python3 -m pytest tests/test_pbo_wf.py -q
EXPECT: all passed
EVIDENCE: 14 passed

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
