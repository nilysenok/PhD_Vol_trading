# GATES — Ж-029 Н6: дневной ряд до 18:50, kodir, 27.10

## G1 — тесты ряда проходят (подложенный вечерний бар, мутации, совпадение до 19.06.2020, close после 22.06.2020)
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_daily_1850.py -q
EXPECT: passed, 0 failed
EVIDENCE: 11 passed in 0.77s (код возврата 0)

## G2 — мутация «граница ≤ 23:59» ловится
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_daily_1850.py -q -k "mutation"
EXPECT: 2 passed (тест ожидает AssertionError на мутанте)
EVIDENCE: тесты test_mutation_boundary_2359_is_caught и test_mutation_boundary_1900_is_caught входят в 11 passed из G1 (pytest.raises(AssertionError) на cut_min=24*60 и 19*60)

## G3 — в новом ряду нет дат ≥ 2022
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_daily_1850.py -q -k no_data_from_2022
EXPECT: 1 passed
EVIDENCE: test_no_data_from_2022_in_new_series входит в 11 passed из G1; плюс читатель фильтрует timestamp < 2022-01-01 на чтении

## G4 — 10 прежних эталонов final3 в архиве posle_N6 с SHA256.txt (git mv)
CHECK: cd docs/otchety/arhiv_do_audita/posle_N6 && shasum -a 256 -c SHA256.txt
EXPECT: 10 строк OK
EVIDENCE: 10 строк OK (код возврата 0)

## G5 — test_dev_reproduces_final3_tables зелёный на новых эталонах
CHECK: FASTBT_MAIN=<копия с симлинками данных> python3 -m pytest tests/test_s9_scope.py::test_dev_reproduces_final3_tables -q
EXPECT: 1 passed
EVIDENCE: 1 passed in 11.37s; весь tests/test_s9_scope.py — 20 passed, 1 skipped; tests/audit_fastbt — 133 passed, 1 skipped

## G6 — fastbt.data читает новый ряд, шапка ≤ 2021
CHECK: FASTBT_MAIN=$PWD python3 -c "from fastbt import data; d=data.daily_df(); print(d.date.max(), data.OHLCV.name)"
EXPECT: 2021-12-30 ohlcv_daily_1850.parquet
EVIDENCE: 2021-12-30 00:00:00 ohlcv_daily_1850.parquet (запуск из файла-обёртки с тем же кодом)

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 6 из 6; ABANDON: 0; pending: 0
