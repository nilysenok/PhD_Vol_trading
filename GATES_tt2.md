# GATES — расследование «стоп бара t знает бар t» (tt=2 и др.), kodir, 29.10.2026

## G1 — тест-воспроизведение проходит на текущем коде
CHECK: python3 -m pytest -q tests/test_rassl_tt2_zaglyad.py
EXPECT: все тесты passed, 0 failed
EVIDENCE: 15 passed in 1.14s (после merge origin/kanon-v22-pereschet @ d66ee36b)

## G2 — тест причинности ловит подложенное заглядывание (high/close бара u+1 в стопе бара u)
CHECK: python3 -m pytest -q tests/test_rassl_tt2_zaglyad.py -k mutaciya
EXPECT: passed (мутант на tt 0/2/3 и на безубытке даёт нарушения > 0)
EVIDENCE: 4 passed, 11 deselected in 0.76s; первая версия теста (3 среза) мутации не ловила — 0 нарушений, срезы заменены на каждый бар

## G3 — соседние тесты движка не сломаны (код не менялся)
CHECK: python3 -m pytest -q tests/audit_fastbt/test_15_gap.py tests/audit_fastbt/test_18_trail_exec.py tests/test_vyhody_v5.py
EXPECT: 0 failed
EVIDENCE: 124 passed in 0.37s

## G4 — оценка Δ Шарпа S3/S4 (≤ 2021) «стоп по t−1» против текущего посчитана
CHECK: FASTBT_MAIN=<worktree> FASTBT_DATA_ROOT=<main>/data/fastbt/daily python3 <scratchpad>/ocenka_tt.py > <scratchpad>/ocenka_tt.txt
EXPECT: таблица Δ по S3/S4, 17 тикеров, без ошибок; даты ряда ≤ 2021-12-31
EVIDENCE: 34 строки tek=/t-1=; «годы (2016, 2017, 2018, 2019) comm 0.0005 даты ≤ 2021-12-30»; S3 медиана Δ +0.063, портфель Δ −0.085; S4 медиана Δ −0.009, портфель Δ +0.131; самопроверка варианта [(1, 2), (1, 3)]

## G5 — отчёт написан, кода fastbt/ в диффе нет
CHECK: git diff --stat origin/kanon-v22-pereschet -- fastbt ; test -f docs/otchety/2026-10-29_rassl_tt2_zaglyad.md
EXPECT: пусто; код возврата 0
EVIDENCE: пусто; rc=0

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
