# GATES — T2 (семья E): исполнение, оценщик sem_e, сетки C/E; кодер, 27.10

Основа: `docs/zadaniya/2026-10-27_semya_E.md` §2, §4, §8; `docs/zadaniya/2026-10-27_komandy_T1_T4.md` T2; контракт `sweep_evaluate.md` v1.1. База ветки: `kanon-v22-pereschet` (ветка воркера стартовала с `main` без `fastbt/exec_e.py`; слита fast-forward).

Сверка exec_e (#290) с §2: лимит, гэпы, тейк, стоп, стоп раньше тейка, стоп в баре входа, временной стоп τ, паузы, ставка внутри дня, Н5 — есть. Нет: (а) сила сигнала и квантильный фильтр q (E0) — §4; (б) тестов на мутации 1, 6, 7, 8 в терминах E; (в) короткой стороны (§2 описывает только покупку) — не делается, см. отступления.

## PR1 — exec_e: q-фильтр силы сигнала + мутационные тесты §8

## G1 — q_mask: окно строго прошлое, q=0 пропускает все сигналы, мутации 6/7 падают
CHECK: python3 -m pytest tests/test_exec_e_mut.py -q -k "m6 or m7"
EXPECT: passed, 0 failed
EVIDENCE: 4 passed (тесты m6 ×3, m7 ×1), код 0

## G2 — таблица «мутация → тест → падает» для всех 12 пунктов §8; на мутанте тест падает
CHECK: FASTBT_MAIN=$PWD python3 <scratchpad>/mutate.py $PWD (подмена исходника exec_e в памяти; пункты 11–12 — мутант внутри теста: test_likvidnost_e, test_metriki_e)
EXPECT: 10 мутаций exec_e: тест на мутанте падает; на исходном 41 passed; пункты 11, 12 — тесты проходят
EVIDENCE: «поймано 10 из 10»; исходник 41 passed; test_likvidnost_e + test_metriki_e: 60 passed, 1 skipped (реальные данные 2017, нет файла) вместе с exec_e. 12 из 12 по таблице PR.

## G3 — существующие тесты exec_e не изменены и проходят
CHECK: git diff --stat origin/kanon-v22-pereschet -- tests/test_exec_e.py tests/test_exec_e_anchor.py; python3 -m pytest tests/test_exec_e.py tests/test_exec_e_anchor.py -q
EXPECT: diff пуст; passed
EVIDENCE: diff пуст; 41 passed

## G4 — файл ≤ 200 строк, ruff/py_compile чисто
CHECK: wc -l fastbt/exec_e.py; python3 -m py_compile fastbt/exec_e.py
EXPECT: ≤ 200; код 0
EVIDENCE: 194 строки; py_compile ok

## PR2 — fastbt/sem_e.py: оценщик C/E под контракт v1.1

## G5 — evaluate(obj, params, period, commission) возвращает ключи контракта (v, sharpe, by_year, sr_d, T, skew, kurt, msr, sortino, mdd, oborot_y, dni_v_rynke, sdelok_y, period, okno_s, okno_po, chestnaya, comm_pct, excess_d)
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "kontrakt"
EXPECT: passed
EVIDENCE: pending

## G6 — oat_sweep и factorial_sweep из sweep_drivers гоняют sem_e на синтетике (без правки sweep_drivers)
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "drivers"; git diff --stat origin/kanon-v22-pereschet -- fastbt/sweep_drivers.py
EXPECT: passed; diff пуст
EVIDENCE: pending

## G7 — E0 при q=0 побайтно равна A (позиции, pnl, fdr)
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "e0_q0"
EXPECT: passed
EVIDENCE: pending

## G8 — ключ кэша = хэш всех входов и кода (Д5): подмена σ, баров, силы, параметра, кода даёт другой ключ
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "kljuch or key"
EXPECT: passed (мутация «ключ без σ» падает)
EVIDENCE: pending

## G9 — период ≥ 2022 и validation не открываются без метки: PermissionError
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "period"
EXPECT: passed
EVIDENCE: pending

## G10 — к2 с лагом 0 равна исходному ряду σ (мутация 8) и сдвиг не нулевой даёт другой Шарп
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "k2"
EXPECT: passed
EVIDENCE: pending

## G11 — проверка работоспособности на реальных данных: один тикер, train 2018–2019, без выбора точек и таблиц
CHECK: python3 -m fastbt.sem_e --smoke (вывод: число строк, нет NaN в excess_d, пути данных < 2022)
EXPECT: код 0; даты ≤ 2019-12-31
EVIDENCE: pending

## PR3 — сетки C/E

## G12 — C_setka_e2.yaml: 324 точки по setka_C_v3.yaml и gl4_parametry_setki
CHECK: python3 -I <scratchpad>/count_setki.py fastbt/designs/sweep/C_setka_e2.yaml
EXPECT: N_C = 324
EVIDENCE: pending

## G13 — E_setka_e2.yaml: E0 4, E1 192, E2 96, E3 ≤ 36, подвыборка u ×2; N задан в файле и в отчёте
CHECK: python3 -I <scratchpad>/count_setki.py fastbt/designs/sweep/E_setka_e2.yaml
EXPECT: E0=4, E1=192, E2=96, E3=36, u=2; N_без_u = 328, N_с_u = 656
EVIDENCE: pending

## G14 — отчёт docs/otchety/2026-10-27_sem_e_setki.md с N до счёта
CHECK: grep -c "N_E" docs/otchety/2026-10-27_sem_e_setki.md
EXPECT: ≥ 1
EVIDENCE: pending

## Общие

## G15 — узкая проверка: fmt/py_compile, cargo xtask check красных 0, git status чист
CHECK: python3 -m py_compile fastbt/exec_e.py fastbt/sem_e.py; cargo xtask check
EXPECT: коды 0
EVIDENCE: pending

## ABANDON
(пусто на старте)

## Реестр
закрыто 0 из 15; ABANDON: 0; pending: 15
