# GATES — T4 (Ж-036, Ж-031 Э4): скрипт открытия теста главы 4, схема чисел, заготовки; arhitektor, 27.10

Владение: `scripts/otkrytie_gl4.py`, `tests/test_otkrytie_gl4*.py`, `docs/final_gl4/zagotovki/`, `docs/final_gl4/chisla_gl4.schema.json`, `docs/kanon/v24/gates/E4_*.md`. Скрипт на реальных данных не запускается; все тесты — синтетика в `tmp_path`.

## G1 — все тесты скрипта открытия проходят
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q
EXPECT: N passed, 0 failed (код 0)
EVIDENCE: 48 passed in 2.37s (код 0), 27.10 ~19:50 по Маку

## G2 — отказ без строки «да» владельца с датой (нет строки / без даты / строки нет в журнале / «да» раньше заморозки / спецификация правлена после «да»)
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k bez_da
EXPECT: все отобранные passed; замок не создан
EVIDENCE: 10 passed, 38 deselected (каждый тест проверяет `not lock.exists()`)

## G3 — отказ при подмене хэша (код, вход, FINAL-B, файл гейтов), до замка
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k podmena
EXPECT: все отобранные passed; замок не создан
EVIDENCE: 9 passed, 39 deselected (скрипт, оценщик, daily_1850, FINAL-B, файл гейтов, вход σ, 10-минутки, новый файл fastbt, исполняемый модуль без SHA)

## G4 — замок ставится до чтения данных теста
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k zamok_do_chteniya
EXPECT: passed (каждое чтение ряда теста видит замок со статусом «начато»)
EVIDENCE: 1 passed, 47 deselected (перехвачены pyarrow.parquet.read_table и pandas.read_parquet; хэш до замка читает байты, значения не разбирает — как final_test_r2.py и final_test_ch4_v2.py)

## G5 — повторный запуск запрещён
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k povtor
EXPECT: passed (второй запуск — СТОП, код 2, выход первого не тронут)
EVIDENCE: 3 passed, 45 deselected (после «завершено», после «прервано», O_EXCL)

## G6 — отказ при незакрытых гейтах (pending, ABANDON, нет файла)
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k geity
EXPECT: passed
EVIDENCE: 3 passed, 45 deselected

## G7 — комиссии ровно {0,035; 0,04}, к2/к3/лестница σ — из спецификации; все задания одним прогоном, без выбора
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k "komissii or vse_zadaniya"
EXPECT: passed
EVIDENCE: 12 passed, 36 deselected

## G8 — JSON чисел синтетического прогона проходит схему `chisla_gl4.schema.json`; схема — валидный JSON с `"v"`
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k shema
EXPECT: passed
EVIDENCE: 3 passed, 45 deselected (jsonschema в окружении нет — проверка встроенной `proverit_shemu`, новой зависимости нет)

## G9 — заготовки таблиц: шапка со статусом «тест», колонки, ни одной строки чисел; колонки = колонки, которые пишет скрипт
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k zagotovki
EXPECT: passed
EVIDENCE: 2 passed, 46 deselected

## G10 — ряд 2022+ до 18:50 строится функциями `fastbt.daily_1850` (aggregate, read_bars) внутри скрипта, под замком
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k ryad_1850
EXPECT: passed (вечерний бар не входит; ряд строится после замка)
EVIDENCE: 1 passed, 47 deselected (close = бар 18:40, бар 19:00 с ценой ×5 не вошёл; даты ≤ okno_testa.po; прогрев из истории)

## G11 — мутации скрипта ловятся (в копии в scratchpad, не в рабочем дереве): (м1) замок после чтения, (м2) без проверки «да», (м3) без сверки SHA кода, (м4) замок без O_EXCL и без проверки существования
CHECK: для каждой мутации: python3 -m pytest <копия>/tests/test_otkrytie_gl4.py -q; test $? -eq 1
EXPECT: 4 из 4 мутантов — pytest вернул 1
EVIDENCE: копия scratchpad/mut (fastbt — ссылка, скрипт подменялся мутантом): база 48 passed; м1 — 1 failed (test_zamok_do_chteniya); м2 — 8 failed (test_bez_da_*); м3 — 2 failed (podmena[scripts/otkrytie_gl4.py], shema_polnyy); м4 — 3 failed (test_povtor_*); у всех pytest=1. 4 из 4

## G12 — изменены только файлы владения
CHECK: git diff --name-only kanon-v22-pereschet...HEAD
EXPECT: только пути из «Владение»
EVIDENCE: 10 путей: docs/final_gl4/chisla_gl4.schema.json, docs/final_gl4/zagotovki/ (5 CSV + spec_otkrytiya.shablon.json), docs/kanon/v24/gates/E4_otkrytie.md, scripts/otkrytie_gl4.py, tests/test_otkrytie_gl4.py

## G13 — PR в `kanon-v22-pereschet` открыт, не слит
CHECK: gh pr view 313 --json baseRefName,state
EXPECT: baseRefName = kanon-v22-pereschet, state = OPEN
EVIDENCE: {"baseRefName":"kanon-v22-pereschet","state":"OPEN"} — PR #313

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 13 из 13; ABANDON: 0; pending: 0
