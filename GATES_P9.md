# GATES — П9: контроли к1–к3 и две проверки процедуры v5, kodir, 29.10

## G1 — тесты листа зелёные
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q
EXPECT: 0 failed
EVIDENCE: 12 passed in 1.37s

## G2 — к2 со сдвигом 0 = исходный (№ 9); к3 обнуляет ровно названную ось (№ 10)
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "lag0 or k3"
EXPECT: passed, 0 failed
EVIDENCE: 2 passed

## G3 — p по перестановкам: равномерно при независимых весах, мал при известном ответе, велик при перепутанном знаке
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "p_perm"
EXPECT: passed, 0 failed
EVIDENCE: 3 passed

## G4 — мутации (окно с годом t; лаг < 21) ловятся; плато → пик — test_plato_not_peak_and_edge_excluded
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "mutatsiya or plato"
EXPECT: passed, 0 failed
EVIDENCE: mutatsiya 2 passed; весь файл 12 passed

## G5 — файлы ≤ 200 строк, py_compile
CHECK: wc -l fastbt/kontroli_v5.py tests/test_kontroli_v5.py && python3 -m py_compile fastbt/kontroli_v5.py
EXPECT: ≤ 200 у каждого, код 0
EVIDENCE: 172 и 174 строки, py_compile 0

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
