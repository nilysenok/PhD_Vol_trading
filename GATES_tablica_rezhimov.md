# GATES — таблица дат режимов v3 (Ж-159 п. 7, Ж-161 п. 1), kodir, 31.10.2026

## G0 — тест падает на базе (модуля нет)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q (первый коммит 2a3d41a4)
EXPECT: ошибка импорта fastbt.tablica_rezhimov_v3, код ≠ 0
EVIDENCE: ImportError: cannot import name 'tablica_rezhimov_v3' from 'fastbt'; 1 error in 0.33s — закрыт

## G1 — синтетика с известным ответом (рост / падение / боковик, net_ma, лаг d−1)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k izvestnyj
EXPECT: passed
EVIDENCE: passed (в составе 19 passed, G6) — закрыт

## G2 — мутации: лаг 0; MA50/MA200 переставлены — тест известного ответа падает
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k mutaciya
EXPECT: passed (мутации ловятся)
EVIDENCE: passed (lag_nol, ma50_ma200_perestavleny, granitsa_snyata) — закрыт

## G3 — совпадение с fazy._priznaki на синтетике (те же даты, тот же режим)
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k fazy
EXPECT: passed
EVIDENCE: passed (900 дней, все три режима встречаются) — закрыт

## G4 — граница: --do 2024-12-31, хвост файла до 2026 → ни чтения, ни строк ≥ 2025; мутация «граница снята» ловится
CHECK: python3 -m pytest tests/test_tablica_rezhimov_v3.py -q -k granitsa
EXPECT: passed
EVIDENCE: passed (max чтения < 2025-01-01; --do 2025-06-30 → PermissionError, файла нет; мутация zona.end → 2026-03 ловится) — закрыт

## G5 — реальный прогон 26.08.2014…31.12.2021 на Маке: sha256 CSV и дни по режимам по годам
CHECK: FASTBT_MAIN=<копия> python3 -m fastbt.tablica_rezhimov_v3 --do 2021-12-31 --out <csv в папке сессии>
EXPECT: печать sha256 и таблицы год × режим; в CSV нет дат ≥ 2022
EVIDENCE: sha256 ddf4710a1bbb10454f4cd2754e6928f02a1dce39cbd3ad0eb18e04b34f60e6e5; 1855 дней, net_ma 200, 2014-08-26…2021-12-30 — закрыт

## G6 — узкий набор
CHECK: python3 -m pytest tests/test_fazy*.py tests/test_tablica_rezhimov_v3.py -q
EXPECT: passed
EVIDENCE: 19 passed in 1.25s — закрыт

## Реестр
закрыто 7 из 7; ABANDON: 0; pending: 0
