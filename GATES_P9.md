# GATES — П9: контроли к1–к3 и две проверки процедуры v5, kodir, 29.10

## G1 — тесты листа зелёные
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q
EXPECT: 0 failed
EVIDENCE: pending

## G2 — к2 со сдвигом 0 = исходный (№ 9); к3 обнуляет ровно названную ось (№ 10)
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "lag0 or k3"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G3 — p по перестановкам: равномерно при шуме, мал при известном ответе, велик при перепутанном знаке
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "p_"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G4 — мутации (окно с годом t; лаг < 21; плато → пик) ловятся
CHECK: python3 -m pytest tests/test_kontroli_v5.py -q -k "mutatsiya"
EXPECT: passed, 0 failed
EVIDENCE: pending

## G5 — файлы ≤ 200 строк, fmt/py_compile
CHECK: wc -l fastbt/kontroli_v5.py tests/test_kontroli_v5.py && python3 -m py_compile fastbt/kontroli_v5.py
EXPECT: ≤ 200 у каждого, код 0
EVIDENCE: pending

## Реестр
закрыто 0 из 5; ABANDON: 0; pending: 5
