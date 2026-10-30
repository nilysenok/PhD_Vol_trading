# Гейты: тестировщик #717

G1 два теста обновлены, проходят
- CHECK: NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_istochniki_v6_dannye.py -p no:cacheprovider
- EXPECT: 4 passed
- EVIDENCE: 4 passed, 4 warnings in 14.73s

G2 мутации а–е пойманы (в отдельной копии mut717)
- CHECK: по одной в копии, pytest tests/test_div_svod.py tests/test_div_svod_mut.py
- EXPECT: на каждой мутации красный
- EVIDENCE: а 3 failed; б 3 failed; в 2 failed; г 4 failed (только новые тесты); д 3 failed; е 6 failed

G3 семейство зелёное
- CHECK: NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_div_svod*.py tests/*v6*.py tests/test_etap1_rezhim_sobytiya.py tests/test_sobytiya_okna_v6.py tests/test_sobytiya_pit.py -p no:cacheprovider
- EXPECT: passed, без failed/error
- EVIDENCE: 923 passed, 125 skipped in 87.02s
