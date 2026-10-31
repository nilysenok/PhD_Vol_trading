# GATES — таблица дат режимов v3 (Ж-159 п. 7, Ж-161 п. 1), kodir, 31.10.2026

## G0 — тест падает на базе (модуля нет)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q (первый коммит)
EXPECT: ошибка импорта fastbt.tablica_rezhimov_v3, код ≠ 0
EVIDENCE: pending

## G1 — синтетика с известным ответом (рост / падение / боковик, net_ma, лаг d−1)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k izvestnyj
EXPECT: passed
EVIDENCE: pending

## G2 — мутации: лаг 0; MA50/MA200 переставлены — тест известного ответа падает
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k mutaciya
EXPECT: passed (мутации ловятся)
EVIDENCE: pending

## G3 — совпадение с fazy._priznaki на синтетике (те же даты, тот же режим)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k fazy
EXPECT: passed
EVIDENCE: pending

## G4 — граница: --do 2024-12-31, хвост файла до 2026 → ни чтения, ни строк ≥ 2025; мутация «граница снята» ловится
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k granitsa
EXPECT: passed
EVIDENCE: pending

## G5 — реальный прогон 26.08.2014…31.12.2021 на Маке: sha256 CSV и дни по режимам по годам
CHECK: python3 -m fastbt.tablica_rezhimov_v3 --do 2021-12-31 --out <csv в папке сессии>
EXPECT: печать sha256 и таблицы год × режим; в CSV нет дат ≥ 2022
EVIDENCE: pending

## G6 — узкий набор
CHECK: python3 -m pytest tests/test_fazy*.py tests/test_tablica_rezhimov_v3.py -q
EXPECT: passed
EVIDENCE: pending

## Реестр
закрыто 0 из 7; ABANDON: 0; pending: 7
