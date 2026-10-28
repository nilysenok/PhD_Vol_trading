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
EVIDENCE: 2 passed. Отступление: сигнатура evaluate(obj, params, period, commission, inp) — пятый аргумент inp (данные), замыкание для драйверов — make_evaluate(obj, inp, commission)

## G6 — oat_sweep и factorial_sweep из sweep_drivers гоняют sem_e на синтетике (без правки sweep_drivers)
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "drivers"; git diff --stat origin/kanon-v22-pereschet -- fastbt/sweep_drivers.py
EXPECT: passed; diff пуст
EVIDENCE: 1 passed (oat n_trials 5, factorial 4); diff пуст

## G7 — E0 при q=0 побайтно равна A (позиции, pnl, fdr)
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "e0_q0"
EXPECT: passed
EVIDENCE: 1 passed. Оговорка: на уровне пары побайтно (test_exec_e_mut::test_m6, tobytes по gross/cost/pos/free/fdr); на уровне портфеля — allclose 1e-12 к msp.portfolio (порядок усреднения), не побайтно

## G8 — ключ кэша = хэш всех входов и кода (Д5): подмена σ, баров, силы, параметра, кода даёт другой ключ
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "kljuch or key"
EXPECT: passed (мутация «ключ без σ» падает)
EVIDENCE: 1 passed; мутации D5 (без σ̂), D5b (без баров), D5c (без obj), P5 (маска без q в ключе) — падают (19 из 19, см. PR)

## G9 — период ≥ 2022 и validation не открываются без метки: PermissionError
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "period"
EXPECT: passed
EVIDENCE: 1 passed; мутации P2 (без замка validation), P4 (даты 2022 допустимы), P1 (прогрев в PnL), P3 (bfill ставки) — падают

## G10 — к2 с лагом 0 равна исходному ряду σ (мутация 8) и сдвиг не нулевой даёт другой Шарп
CHECK: python3 -m pytest tests/test_sem_e.py -q -k "k2"
EXPECT: passed
EVIDENCE: 1 passed; мутация 8 (лаг смещён на 1) падает в test_k2_lag0... и в test_exec_e_mut::test_m8

## G11 — проверка работоспособности на реальных данных: один тикер, train 2018–2019, без выбора точек и таблиц
CHECK: FASTBT_MAIN=$PWD python3 <scratchpad>/smoke.py $PWD SBER (скрипт вне репозитория: A из кэша s9 dev, 5-мин бары bars5m, σ̂ naive/const из sigma_e; данные подключены симлинками на чтение, после прогона сняты)
EXPECT: код 0; даты ≤ 2019-12-31; NaN в excess_d = 0
EVIDENCE: даты 2018-01-03…2019-12-30 (506 дн.), пустых дней 0; E/naive, E/const, C/naive: T 506, NaN 0, окно 2018-01-01…2019-12-31. Метрики не печатались и не сохранялись (выбора точек нет). q > 0 на реальных данных не гонялась: см. ABANDON G11b

## PR3 — сетки C/E

## G12 — C_setka_e2.yaml: 324 точки по setka_C_v3.yaml и gl4_parametry_setki
CHECK: python3 <scratchpad>/count_setki.py fastbt/designs/sweep/C_setka_e2.yaml (без -I: PyYAML в пользовательском site-packages); python3 -m pytest tests/test_sem_e_setki.py -q
EXPECT: N_C = 324
EVIDENCE: hyb 144, harj 144, naive 36, N_C = 324, код 0; тест N из осей passed

## G13 — E_setka_e2.yaml: E0 4, E1 192, E2 96, E3 ≤ 36, подвыборка u ×2; N задан в файле и в отчёте
CHECK: python3 <scratchpad>/count_setki.py fastbt/designs/sweep/E_setka_e2.yaml
EXPECT: E0=4, E1=192, E2=96, E3=36, u=2; N_без_u = 328, N_с_u = 656
EVIDENCE: {'E0': 4, 'E1': 192, 'E2': 96, 'E3': 36}, N_без_u = 328, u = 2, N_с_u = 656, код 0; все значения осей E0/E1/E2 и C принимаются sem_e (tests/test_sem_e_setki.py, 19 passed вместе с test_sem_e)

## G14 — отчёт docs/otchety/2026-10-27_sem_e_setki.md с N до счёта
CHECK: grep -c "N_E" docs/otchety/2026-10-27_sem_e_setki.md
EXPECT: ≥ 1
EVIDENCE: 2 (строк с N_E)

## Общие

## G15 — узкая проверка: py_compile, весь набор T2 проходит, git status чист
CHECK: python3 -m py_compile fastbt/exec_e.py fastbt/sem_e.py; FASTBT_MAIN=$PWD python3 -m pytest tests/test_exec_e.py tests/test_exec_e_anchor.py tests/test_exec_e_mut.py tests/test_sem_e.py tests/test_sem_e_setki.py -q
EXPECT: коды 0
EVIDENCE: py_compile ок; 60 passed; мутации 21 из 21 падают (скрипт вне репозитория)

## ABANDON
ABANDON: G11b q > 0 на реальных данных — сила сигнала ансамбля A (голос 10 конфигураций) нигде не хранится: в кэше s9 только позиции; постановка §4 ⚑ «определение из кода A». Модуль требует Pair.strength и при q > 0 без неё падает ValueError («нет силы сигнала»), тихой подмены нет. Нужно решение: определение силы для ансамбля (❓ владельцу/T1, класс A, так как меняет точки E0).

ABANDON: G15b `cargo xtask check` — в этом репозитории (Python, база `kanon-v22-pereschet`) команды xtask нет (`error: no such command: xtask`); CLAUDE.md проекта: «не применяется». Замена — py_compile и pytest в G15.

## Правки по вердикту #311 (координатор, 27.10)
- Шорты A: пара_e пропускает короткую сторону как в A (с подвыборкой u); тест `test_e0_q0_equals_full_a_with_shorts` (шорты ≥ 10 % дней, позиции побайтно, pnl allclose 1e-12 к msp.portfolio); мутация «шорты выброшены» падает.
- Сделки и оборот считаются только в окне периода (без прогрева 2017); для E сторона = единица позиции, поэтому oborot_y = sdelok_y; для C они различны (разворот 2 против 1, шаги 0,5); тесты и мутации W1, W2 падают.
- Д5: Series σ̂ — хэшируются значения и индекс; мутации D5, D5s падают.
- test_m6 сверяет с независимым эталоном msp.portfolio, не с sig.
- Сетка C: строка `baza: A_prim` (baza: true, в N не входит), N_C = 324.
- Мутации: 25 из 25 (скрипт вне репозитория). Тесты: 62 passed (только перечисленные файлы).

## Реестр
закрыто 15 из 15 (G1–G15); ABANDON: 2 (G11b — сила A для q > 0 на реальных данных; G15b — xtask отсутствует); pending: 0
Отчёт — «передача», а не «готово» (polnota.md п. 12): есть ABANDON.
