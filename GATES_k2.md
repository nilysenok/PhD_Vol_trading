# GATES — К-2 досчёт членов калибратора v5 на сервере, kodir, 29.10

## G1 — соседи фазы 2 пишутся в ledger и в N, повтор не дублирует строку
CHECK: python3 -m pytest tests/test_kalibr_v5_doschet.py -q -k "sosedi"
EXPECT: passed
EVIDENCE: 4 passed, 9 deselected

## G2 — голос = execlayer.vote на бинарных знаковых позициях, ≤ 1e-12; знак из expo − 2·short
CHECK: python3 -m pytest tests/test_kalibr_v5_doschet.py -q -k "golos or znak"
EXPECT: passed
EVIDENCE: 4 passed, 9 deselected

## G3 — ряды членов для к2 и «одной точки» (срезы года)
CHECK: python3 -m pytest tests/test_kalibr_v5_doschet.py -q -k "ryady"
EXPECT: passed
EVIDENCE: 1 passed, 12 deselected

## G4 — отказы: kod/vhody ≠ ledger, даты ≥ 2020
CHECK: python3 -m pytest tests/test_kalibr_v5_doschet.py -q -k "otkaz"
EXPECT: passed
EVIDENCE: 2 passed, 11 deselected

## G5 — мутации ловятся (нет записи соседа; знак без short)
CHECK: python3 -m pytest tests/test_kalibr_v5_doschet.py -q -k "mutaciya"
EXPECT: passed
EVIDENCE: 2 passed, 11 deselected

## G6 — стиль и канон
CHECK: python3 -m py_compile fastbt/kalibr_v5_doschet.py && wc -l fastbt/kalibr_v5_doschet.py && pytest tests/test_kalibr_v5.py
EXPECT: ≤ 200 строк; старые тесты passed
EVIDENCE: 118 строк модуль, 200 тест; test_kalibr_v5.py passed. test_kalibr_v5_golos_v2.py: 6 failed ДО моего файла тоже (flt6–8, 12–14) — не моё, в отчёте.

## Реестр
закрыто 6 из 6; ABANDON: 0; pending: 0
