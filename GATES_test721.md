# GATES — тестировщик #721 (план Морриса v6)

## G1 — мутации а–ж прогнаны в отдельной копии (scratchpad/mut721), каждая ловится тестами
CHECK: таблица в описании PR; в копии по одной правке, откат cp
EXPECT: каждая мутация роняет хотя бы один тест
EVIDENCE: 11 вариантов мутаций из 11 пойманы (7 — только новыми тестами, 4 — обоими наборами)

## G2 — тесты tests/test_morris_v6_plan_mut.py проходят на коде PR
CHECK: NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_morris_v6_plan_mut.py -p no:cacheprovider
EXPECT: 17 passed
EVIDENCE: 17 passed in 1.22s

## G3 — свойство на 6 реальных кубах v6 (соседние точки — одна ось, эффект на ней)
CHECK: pytest -k real_kub
EXPECT: 6 passed
EVIDENCE: входит в 17 passed

## G4 — узкий набор *morris*, *v6*
CHECK: NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/*morris*.py tests/*v6*.py -p no:cacheprovider
EXPECT: passed, без failed/error
EVIDENCE: 988 passed, 125 skipped, 7 warnings in 137.29s (код 0)
